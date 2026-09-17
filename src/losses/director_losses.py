# src/losses/director_losses.py
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


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
) -> Dict[str, torch.Tensor]:
    """Render support-weighted Gaussian target heatmaps (Equation 12 in paper).

    Returns:
      Y1: (B, 1, H, W) - Top-1 primary mode heatmap target Y_t^(1)
      Y_minus: (B, 1, H, W) - Top-2+ auxiliary modes heatmap target Y_t^-
      Y_all: (B, 1, H, W) - Joint target Y_t = max(Y1, Y_minus)
      mask_omega: (B, 1, H, W) - Binary mask for auxiliary region Omega_t = {q: A_t^(1)(q) == 0}
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

        # Top-1 primary mode (k=1)
        c1_y, c1_x = c_tensor[0, 0], c_tensor[0, 1]
        w1 = (s_tensor[0] / float(u_observers)).clamp(0.0, 1.0)
        ct_top1[b, 0] = c1_x
        ct_top1[b, 1] = c1_y

        feat_c1_x = int(torch.clamp(c1_x / stride, 0, feat_w - 1).item())
        feat_c1_y = int(torch.clamp(c1_y / stride, 0, feat_h - 1).item())
        pos_mask_top1[b, 0, feat_c1_y, feat_c1_x] = 1.0

        dist_sq1 = (grid_x - c1_x) ** 2 + (grid_y - c1_y) ** 2
        Y1[b, 0] = w1 * torch.exp(-dist_sq1 / two_sig_sq)

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
                dist_sq_k = (grid_x - ck_x) ** 2 + (grid_y - ck_y) ** 2
                gauss_k = wk * torch.exp(-dist_sq_k / two_sig_sq)
                aux_accum = torch.maximum(aux_accum, gauss_k)

                feat_ck_x = int(torch.clamp(ck_x / stride, 0, feat_w - 1).item())
                feat_ck_y = int(torch.clamp(ck_y / stride, 0, feat_h - 1).item())
                b_boxes.append(torch.tensor([ck_x, ck_y, float(vp_w), float(vp_h)], device=device))
                b_inds.append(feat_ck_y * feat_w + feat_ck_x)

            Y_minus[b, 0] = aux_accum

        all_target_boxes.append(torch.stack(b_boxes, dim=0) if b_boxes else torch.zeros((0, 4), device=device))
        all_target_inds.append(torch.tensor(b_inds, dtype=torch.long, device=device) if b_inds else torch.zeros((0,), dtype=torch.long, device=device))

    Y_all = torch.maximum(Y1, Y_minus)

    return {
        "Y1": Y1,
        "Y_minus": Y_minus,
        "Y_all": Y_all,
        "mask_omega": mask_omega,
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
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """CornerNet modified focal loss for Top-1 Human Consensus (Eq 13, 14).

    ℓ_t(q) =
      (1 - Ŷ_t)^α * log(Ŷ_t),                           if pos_mask(q) == 1
      (1 - Y_t^(1))^β * Ŷ_t^α * log(1 - Ŷ_t),           otherwise
    """
    pred = pred_hm.clamp(eps, 1.0 - eps)

    pos_term = torch.pow(1.0 - pred, alpha) * torch.log(pred) * pos_mask
    neg_term = (
        torch.pow(1.0 - target_hm_top1, beta)
        * torch.pow(pred, alpha)
        * torch.log(1.0 - pred)
        * (1.0 - pos_mask)
    )

    num_pos = pos_mask.sum().clamp_min(1.0)
    loss = -(pos_term.sum() + neg_term.sum()) / num_pos
    return loss


def ranked_mode_coverage_loss(
    pred_hm: torch.Tensor,
    target_hm_minus: torch.Tensor,
    mask_omega: torch.Tensor,
    has_aux: torch.Tensor,
) -> torch.Tensor:
    """Masked MSE on auxiliary modes over Omega_t (Eq 15, 16).

    If M_t^- is empty (has_aux is False), loss is exactly 0.0.
    """
    if not has_aux.any():
        return torch.tensor(0.0, device=pred_hm.device, dtype=pred_hm.dtype)

    # Compute masked squared error per sample in batch
    diff_sq = (pred_hm - target_hm_minus) ** 2
    masked_diff = diff_sq * mask_omega  # (B, 1, H, W)

    sample_losses = []
    for b in range(pred_hm.size(0)):
        if not has_aux[b]:
            continue
        omega_area = mask_omega[b].sum().clamp_min(1.0)
        s_loss = masked_diff[b].sum() / omega_area
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
) -> torch.Tensor:
    """Trajectory Smoothness loss (Eq 18): L2 displacement between primary centers."""
    diff = primary_centers_t - primary_centers_next  # (B, 2)
    disp_sq = torch.sum(diff ** 2, dim=-1)  # (B,)

    if valid_mask is not None:
        valid = valid_mask & (~torch.isnan(disp_sq))
        if not valid.any():
            return torch.tensor(0.0, device=primary_centers_t.device, dtype=torch.float32)
        return disp_sq[valid].mean()

    valid = ~torch.isnan(disp_sq)
    if not valid.any():
        return torch.tensor(0.0, device=primary_centers_t.device, dtype=torch.float32)
    return disp_sq[valid].mean()
