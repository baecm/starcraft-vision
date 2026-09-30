# src/losses/director_losses.py
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def _add_observer_positives(
    Y1: torch.Tensor,
    pos_mask: torch.Tensor,
    aux_support: torch.Tensor,
    obs_boxes_list: List[Optional[torch.Tensor]],
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    stride: int,
    feat_h: int,
    feat_w: int,
    two_sig_sq: float,
) -> None:
    """Mark each observer's own viewport centre as a primary positive, in place.

    L_hcm otherwise has exactly one positive cell per frame - the Top-1 mode -
    against roughly a thousand negatives, so the localisation signal is a
    single "yes" per image. The vanilla CenterNet baseline in this repo instead
    places a positive at every observer viewport centre, and it reaches a
    higher IR than Director despite a four-conv backbone with no pretraining
    and no FPN. That is the closest controlled comparison available: same task,
    same data, same metric, much weaker encoder.

    It is also the right target for IR, which measures overlap with the
    *union* of observer viewports. Landing on any observer's viewport scores
    well, so supervising only the consensus mode optimises something narrower
    than what is measured.

    Centres falling inside the auxiliary support are skipped, since L_rmc owns
    those cells and regresses them to support/U; making them amplitude-1
    positives would put the two objectives back in conflict.
    """
    for b, boxes in enumerate(obs_boxes_list):
        if boxes is None or len(boxes) == 0:
            continue
        boxes = boxes.to(device=Y1.device, dtype=Y1.dtype)
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
        fx = torch.clamp((cx / stride).long(), 0, feat_w - 1)
        fy = torch.clamp((cy / stride).long(), 0, feat_h - 1)

        free = aux_support[b, 0, fy, fx] == 0
        fx, fy = fx[free], fy[free]
        if fx.numel() == 0:
            continue

        pos_mask[b, 0, fy, fx] = 1.0
        for cell_y, cell_x in zip(fy.tolist(), fx.tolist()):
            dist_sq = (grid_x - cell_x * stride) ** 2 + (grid_y - cell_y * stride) ** 2
            Y1[b, 0] = torch.maximum(Y1[b, 0], torch.exp(-dist_sq / two_sig_sq))


def render_gaussian_heatmap_targets(
    modes_list: List[Dict[str, torch.Tensor]],
    batch_size: int,
    feat_h: int,
    feat_w: int,
    stride: int,
    device: torch.device,
    render_sigma: float = config.DIRECTOR_RENDER_SIGMA,
    u_observers: int = config.NUM_OBSERVERS_U,
    viewport_size_hw: Tuple[int, int] = config.VIEWPORT_SIZE_HW,
    obs_boxes_list: Optional[List[Optional[torch.Tensor]]] = None,
) -> Dict[str, torch.Tensor]:
    """Render support-weighted Gaussian target heatmaps (Equation 12 in paper).

    Returns:
      Y1: (B, 1, H, W) - Top-1 primary mode heatmap target Y_t^(1)
      Y_minus: (B, 1, H, W) - Top-2+ auxiliary modes heatmap target Y_t^-
      Y_all: (B, 1, H, W) - Joint target Y_t = max(Y1, Y_minus)
      mask_omega: (B, 1, H, W) - Binary mask for auxiliary region Omega_t = {q: A_t^(1)(q) == 0}
      aux_support: (B, 1, H, W) - Binary mask of cells carrying auxiliary mode
                   mass inside Omega_t. This is L_rmc's domain, and the region
                   L_hcm must ignore rather than treat as background.
      pos_mask_top1: (B, 1, H, W) - Binary indicator at Top-1 center grid cell
      ct_top1: (B, 2) - Top-1 center coordinate (x, y) in tile units (or nan if none)
      has_aux: (B,) - Boolean tensor indicating whether M_t^- is non-empty
      target_boxes: List of (M, 4) boxes for offset/size regression
      target_inds: List of (M,) 1D grid indices
    """
    Y1 = torch.zeros((batch_size, 1, feat_h, feat_w), device=device)
    Y_minus = torch.zeros((batch_size, 1, feat_h, feat_w), device=device)
    mask_omega = torch.ones((batch_size, 1, feat_h, feat_w), device=device)
    pos_mask_top1 = torch.zeros((batch_size, 1, feat_h, feat_w), device=device)
    ct_top1 = torch.full((batch_size, 2), float("nan"), device=device)
    has_aux = torch.zeros((batch_size,), dtype=torch.bool, device=device)

    # Grid coordinates in tile units: (x, y)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(feat_h, device=device, dtype=torch.float32) * stride,
        torch.arange(feat_w, device=device, dtype=torch.float32) * stride,
        indexing="ij",
    )

    vp_h, vp_w = viewport_size_hw
    two_sig_sq = 2.0 * (render_sigma ** 2)

    all_target_boxes = []
    all_target_inds = []

    for b in range(batch_size):
        if b >= len(modes_list):
            all_target_boxes.append(torch.zeros((0, 4), device=device))
            all_target_inds.append(torch.zeros((0,), dtype=torch.long, device=device))
            continue

        item = modes_list[b]
        centers = item.get("centers")  # (M, 2) [row, col] -> [y, x] in tile coordinates
        supports = item.get("support")  # (M,)

        if centers is None or len(centers) == 0:
            all_target_boxes.append(torch.zeros((0, 4), device=device))
            all_target_inds.append(torch.zeros((0,), dtype=torch.long, device=device))
            continue

        if isinstance(centers, np_type := (torch.Tensor,)):
            c_tensor = centers.to(device=device, dtype=torch.float32)
        else:
            c_tensor = torch.as_tensor(centers, device=device, dtype=torch.float32)

        if isinstance(supports, np_type):
            s_tensor = supports.to(device=device, dtype=torch.float32)
        else:
            s_tensor = torch.as_tensor(supports, device=device, dtype=torch.float32)

        n_modes = len(c_tensor)
        b_boxes = []
        b_inds = []

        # Top-1 primary mode (k=1).
        #
        # The primary renders at full amplitude rather than at support/U. The
        # support weighting exists to rank the modes, but the primary is
        # already separated from the auxiliaries by which loss supervises it -
        # L_hcm owns this cell, L_rmc owns Psi_t - so within the primary the
        # weighting only weakens supervision. It weakens it most on exactly the
        # frames that are hardest: when observers disagree, the Top-1 support
        # is low, so w1 is small, so every cell around the peak carries a
        # penalty-reduction factor (1 - w1*exp(...))^beta close to 1 and is
        # trained as a hard negative. The primary then gets a near-delta target
        # on precisely the multimodal frames this work is about. Ranking is
        # unaffected: auxiliary amplitudes stay at support/U <= 1.
        c1_y, c1_x = c_tensor[0, 0], c_tensor[0, 1]
        w1 = torch.ones((), device=device)
        ct_top1[b, 0] = c1_x
        ct_top1[b, 1] = c1_y

        feat_c1_x = int(torch.clamp(c1_x / stride, 0, feat_w - 1).item())
        feat_c1_y = int(torch.clamp(c1_y / stride, 0, feat_h - 1).item())
        pos_mask_top1[b, 0, feat_c1_y, feat_c1_x] = 1.0

        # Centre the Gaussian on the assigned grid cell, not on the continuous
        # tile coordinate, exactly as CenterNet renders at ct_int. With
        # sigma=2 tiles on a stride-4 grid the nearest sampled point can be
        # 2*sqrt(2) tiles from a continuous centre, which scaled the peak down
        # by exp(-8/8) = 0.37 - so the amplitude encoded quantisation error as
        # much as observer support, defeating the ranking it exists to carry.
        # A mode with support 1 of U=5 now renders at exactly 0.2 instead of
        # somewhere in 0.074-0.2. The sub-cell residual is the offset head's job.
        Y1[b, 0] = w1 * torch.exp(
            -((grid_x - feat_c1_x * stride) ** 2 + (grid_y - feat_c1_y * stride) ** 2) / two_sig_sq
        )

        # Primary region coverage A_t^(1): box [c1_x - w/2, c1_y - h/2, c1_x + w/2, c1_y + h/2]
        x_min = c1_x - vp_w / 2.0
        x_max = c1_x + vp_w / 2.0
        y_min = c1_y - vp_h / 2.0
        y_max = c1_y + vp_h / 2.0

        in_primary = (grid_x >= x_min) & (grid_x <= x_max) & (grid_y >= y_min) & (grid_y <= y_max)
        mask_omega[b, 0] = (~in_primary).float()

        b_boxes.append(torch.tensor([c1_x, c1_y, float(vp_w), float(vp_h)], device=device))
        b_inds.append(feat_c1_y * feat_w + feat_c1_x)

        # Auxiliary modes (k >= 2)
        if n_modes > 1:
            has_aux[b] = True
            aux_accum = torch.zeros((feat_h, feat_w), device=device)
            for k in range(1, n_modes):
                ck_y, ck_x = c_tensor[k, 0], c_tensor[k, 1]
                wk = (s_tensor[k] / float(u_observers)).clamp(0.0, 1.0)
                feat_ck_x = int(torch.clamp(ck_x / stride, 0, feat_w - 1).item())
                feat_ck_y = int(torch.clamp(ck_y / stride, 0, feat_h - 1).item())

                dist_sq_k = (grid_x - feat_ck_x * stride) ** 2 + (grid_y - feat_ck_y * stride) ** 2
                gauss_k = wk * torch.exp(-dist_sq_k / two_sig_sq)
                aux_accum = torch.maximum(aux_accum, gauss_k)

                b_boxes.append(torch.tensor([ck_x, ck_y, float(vp_w), float(vp_h)], device=device))
                b_inds.append(feat_ck_y * feat_w + feat_ck_x)

            Y_minus[b, 0] = aux_accum

        all_target_boxes.append(torch.stack(b_boxes, dim=0) if b_boxes else torch.zeros((0, 4), device=device))
        all_target_inds.append(torch.tensor(b_inds, dtype=torch.long, device=device) if b_inds else torch.zeros((0,), dtype=torch.long, device=device))

    # Cells whose auxiliary mass is worth supervising. Everything below the
    # floor is indistinguishable from background and belongs to L_hcm.
    aux_support = ((Y_minus > config.DIRECTOR_AUX_SUPPORT_FLOOR).float() * mask_omega)

    if obs_boxes_list is not None:
        _add_observer_positives(
            Y1=Y1, pos_mask=pos_mask_top1, aux_support=aux_support,
            obs_boxes_list=obs_boxes_list, grid_x=grid_x, grid_y=grid_y,
            stride=stride, feat_h=feat_h, feat_w=feat_w, two_sig_sq=two_sig_sq,
        )

    Y_all = torch.maximum(Y1, Y_minus)

    return {
        "Y1": Y1,
        "Y_minus": Y_minus,
        "Y_all": Y_all,
        "mask_omega": mask_omega,
        "aux_support": aux_support,
        "pos_mask_top1": pos_mask_top1,
        "ct_top1": ct_top1,
        "has_aux": has_aux,
        "target_boxes": all_target_boxes,
        "target_inds": all_target_inds,
    }


def human_consensus_match_loss(
    pred_hm: torch.Tensor,
    target_hm_top1: torch.Tensor,
    pos_mask: torch.Tensor,
    ignore_mask: Optional[torch.Tensor] = None,
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-6,
    neg_weight: float = 1.0,
) -> torch.Tensor:
    """CornerNet modified focal loss for Top-1 Human Consensus (Eq 13, 14).

    ℓ_t(q) =
      (1 - Ŷ_t)^α * log(Ŷ_t),                           if pos_mask(q) == 1
      (1 - Y_t)^β * Ŷ_t^α * log(1 - Ŷ_t),               if q is supervised here
      0,                                                if ignore_mask(q) == 1

    Two corrections to the published form, both about the auxiliary modes.

    `target_hm_top1` should be handed the *joint* target Y_t, not Y_t^(1). In
    CornerNet the (1 - Y)^beta factor protects every ground-truth location from
    being pulled down as a hard negative. Weighting by Y^(1) alone leaves the
    auxiliary modes unprotected: Y^(1) ~ 0 there, so the factor is ~1 and the
    focal term trains the model to erase exactly what L_rmc is trying to
    create.

    `ignore_mask` then removes the auxiliary support region from this loss
    altogether. Down-weighting is not enough - at an auxiliary peak with
    support 1 of U=5 the target is 0.2, so (1 - 0.2)^4 = 0.41 still leaves
    more downward pressure than L_rmc can answer with. Paper Section 4.3.2
    claims the Omega restriction makes the two objectives "complementary
    rather than adversarial", but Omega excludes only the *primary* region;
    nothing stopped this term from covering Omega at full weight. Splitting
    the domains makes that claim true: L_hcm owns the primary cell and the
    genuine background, L_rmc owns the auxiliary support.

    `neg_weight` keeps the loss comparable across heatmap resolutions. The
    positive term is normalised by num_pos, which is one cell per sample by
    construction here and so does not grow with resolution, while the negative
    term sums over every cell and does. Callers that change the output stride
    pass the per-cell map area relative to the stride the loss weights were
    calibrated at; see DirectorCenterNet._hcm_neg_weight.
    """
    pred = pred_hm.clamp(eps, 1.0 - eps)

    neg_domain = 1.0 - pos_mask
    if ignore_mask is not None:
        neg_domain = neg_domain * (1.0 - ignore_mask)

    pos_term = torch.pow(1.0 - pred, alpha) * torch.log(pred) * pos_mask
    neg_term = (
        torch.pow(1.0 - target_hm_top1, beta)
        * torch.pow(pred, alpha)
        * torch.log(1.0 - pred)
        * neg_domain
    )

    num_pos = pos_mask.sum().clamp_min(1.0)
    loss = -(pos_term.sum() + neg_weight * neg_term.sum()) / num_pos
    return loss


def ranked_mode_coverage_loss(
    pred_hm: torch.Tensor,
    target_hm_minus: torch.Tensor,
    mask_omega: torch.Tensor,
    has_aux: torch.Tensor,
    aux_support: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Masked MSE on auxiliary modes over Omega_t (Eq 15, 16).

    If M_t^- is empty (has_aux is False), loss is exactly 0.0.

    Averaging over all of Omega_t is what made this loss inert. Omega is ~1009
    of the 1024 cells, while the auxiliary mass sits in roughly a dozen of
    them, so dividing by |Omega| diluted the only informative cells by ~10^2
    while L_hcm normalised by num_pos (1 per sample). Measured at an auxiliary
    cell the two differed by ~760x, against the auxiliary regions - which is
    why no ablation could show an effect for lambda_rmc and why n_pred stayed
    at 1.4-1.5 with K=3.

    So the domain is now the auxiliary support within Omega rather than all of
    Omega, and the mean is taken over that. Suppressing the true background is
    L_hcm's job and it already covers those cells; this term only has to raise
    the prediction at the ranked minority modes to their support-weighted
    amplitude. `aux_support` is omitted only by older callers, which fall back
    to deriving it from a positive target.
    """
    if not has_aux.any():
        return torch.tensor(0.0, device=pred_hm.device, dtype=pred_hm.dtype)

    if aux_support is None:
        aux_support = (target_hm_minus > config.DIRECTOR_AUX_SUPPORT_FLOOR).to(pred_hm.dtype) * mask_omega

    diff_sq = (pred_hm - target_hm_minus) ** 2
    masked_diff = diff_sq * aux_support  # (B, 1, H, W)

    sample_losses = []
    for b in range(pred_hm.size(0)):
        if not has_aux[b]:
            continue
        support_area = aux_support[b].sum().clamp_min(1.0)
        s_loss = masked_diff[b].sum() / support_area
        sample_losses.append(s_loss)

    if not sample_losses:
        return torch.tensor(0.0, device=pred_hm.device, dtype=pred_hm.dtype)

    return torch.stack(sample_losses).mean()


def pairwise_box_iou(boxes: torch.Tensor) -> torch.Tensor:
    """Compute differentiable pairwise IoU matrix for boxes of shape (K, 4) [x1, y1, x2, y2]."""
    if boxes.size(0) < 2:
        return torch.zeros((0, 0), device=boxes.device, dtype=boxes.dtype)

    b1 = boxes.unsqueeze(1)  # (K, 1, 4)
    b2 = boxes.unsqueeze(0)  # (1, K, 4)

    inter_x1 = torch.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = torch.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = torch.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = torch.minimum(b1[..., 3], b2[..., 3])

    inter_w = F.relu(inter_x2 - inter_x1)
    inter_h = F.relu(inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union_area = area1 + area2 - inter_area

    iou = inter_area / (union_area + 1e-7)
    return iou


def spatial_repulsion_loss(predicted_boxes_batch: List[torch.Tensor]) -> torch.Tensor:
    """Spatial Repulsion loss (Eq 17): sum of pairwise IoUs between predicted regions."""
    batch_losses = []
    device = None

    for boxes in predicted_boxes_batch:
        if device is None:
            device = boxes.device
        if boxes.size(0) < 2:
            continue
        iou_matrix = pairwise_box_iou(boxes)  # (K, K)
        # Sum upper triangular elements (k < l)
        triu_indices = torch.triu_indices(boxes.size(0), boxes.size(0), offset=1, device=boxes.device)
        pair_ious = iou_matrix[triu_indices[0], triu_indices[1]]
        batch_losses.append(pair_ious.sum())

    if not batch_losses:
        return torch.tensor(0.0, device=device or torch.device("cpu"), dtype=torch.float32)

    return torch.stack(batch_losses).mean()


def trajectory_smoothness_loss(
    primary_centers_t: torch.Tensor,
    primary_centers_next: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    huber_delta: float = config.DIRECTOR_SMOOTH_HUBER_DELTA,
    norm_scale: float = config.MAP_DIAGONAL_TILES,
) -> torch.Tensor:
    """Trajectory Smoothness loss (Eq 18): Huber displacement between primary centers.

    The squared-L2 form this replaces made "hold still" cheaper than "follow the
    action". Displacement is in tiles, so on a 128x128 map two independent
    predictions differ by E[||d||^2] ~ 5461 at initialisation; times lambda_sm
    0.2 that is ~1090 against a focal term of O(1). Because
    `engine_safe.train_one_epoch_safe` clips the *global* gradient norm to
    `config.GRAD_CLIP_NORM`, one term that large does not merely outweigh the
    others - it rescales the whole update vector and leaves them almost no
    share of it. Every ablation carrying L_smooth collapsed to a fixed camera;
    none without it did.

    Two changes, and the order between them is the point:

    1. Huber in *tile* units, so `huber_delta` reads as "5 tiles" and the
       per-sample gradient is capped at delta instead of growing with the
       displacement. Small motion stays quadratic and nearly free.
    2. Divide by the map diagonal afterwards, which is what puts the result on
       an O(1) scale (worst case ~4.9).

    Normalising *first* and then applying Huber would place delta at
    5/181 = 0.028, so the linear branch carries a slope of 0.028 instead of 5
    and the term is divided by 181 a second time: ~10^4 below the unnormalised
    squared form, ~10^2 below what this function returns. Either way it
    switches the objective off rather than fixing it - and would make the
    full-vs-no_smooth ablation a null comparison.
    """
    diff = primary_centers_t - primary_centers_next  # (B, 2)
    disp_sq = torch.sum(diff ** 2, dim=-1)  # (B,)

    # Huber on ||d|| expressed via ||d||^2, so the quadratic branch needs no
    # sqrt at all and the linear branch only ever takes sqrt of something
    # >= delta^2. A plain sqrt(disp_sq) would have an infinite derivative at
    # zero displacement, which is exactly where the frozen-camera solution sits.
    delta_sq = huber_delta ** 2
    quadratic = 0.5 * disp_sq
    linear = huber_delta * torch.sqrt(disp_sq.clamp_min(delta_sq)) - 0.5 * delta_sq
    per_sample = torch.where(disp_sq <= delta_sq, quadratic, linear) / norm_scale

    if valid_mask is not None:
        valid = valid_mask & (~torch.isnan(per_sample))
    else:
        valid = ~torch.isnan(per_sample)

    if not valid.any():
        return torch.tensor(0.0, device=primary_centers_t.device, dtype=torch.float32)
    return per_sample[valid].mean()
