from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

import config
from losses.director_losses import (
    human_consensus_match_loss,
    ranked_mode_coverage_loss,
    render_gaussian_heatmap_targets,
    spatial_repulsion_loss,
    trajectory_smoothness_loss,
)
from utils.logger import Logger

# The FPN level the heads read ('0') is stride 4, and that is the resolution
# every cell-unit knob below was tuned at. An output stride finer than this is
# produced by an upsampling decoder, not by touching the backbone, so this
# stays fixed and `down_ratio` is the *output* stride.
FPN_STRIDE = 4

# Radius, in tiles, that NMS suppresses around a peak. The 3x3 kernel used at
# stride 4 is a radius of 4 tiles, and that is what this preserves, so a finer
# output stride does not silently change the decoder. It stays safely below
# MODE_EXTRACTION_MIN_SEP / 2 = 6 tiles, the largest radius that cannot merge
# two genuine ranked modes; a radius far below it is what lets one mode emit
# several peaks inside its own viewport, so this is the knob to raise if
# n_pred balloons with duplicates at a finer stride.
NMS_RADIUS_TILES = 4.0


def _kaiming_init_conv(conv: nn.Conv2d):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def _init_head(head: nn.Sequential, final_bias: float = 0.0):
    """Kaiming for the hidden convs, near-zero for the output conv.

    fan_out is the wrong mode for a head's final 1x1 conv and catastrophically
    so here: for Conv2d(64, 1, 1) fan_out is 1, giving std = sqrt(2) = 1.414 on
    weights that sum 64 ReLU activations. The heatmap logits came out with a
    standard deviation of roughly 10, sigmoid saturated, and the -2.19 bias
    prior was swamped - scripts/probe_director.py measured L_hcm at 1970 and
    L_off at 19.3 on random input, the latter being an L1 against targets in
    [0, 1).

    That second number is the corner-sticking mechanism from the ablations: an
    offset of ~19 puts the decoded centre (cell + offset) * stride some 76
    tiles from its cell, far outside a 128-tile map, where the box clamp pins
    it to an edge. CenterNet initialises output layers at std 0.001 for exactly
    this reason, which also lets the bias prior actually set the initial
    probability.
    """
    convs = [m for m in head.modules() if isinstance(m, nn.Conv2d)]
    for conv in convs[:-1]:
        _kaiming_init_conv(conv)
    nn.init.normal_(convs[-1].weight, std=0.001)
    if convs[-1].bias is not None:
        nn.init.constant_(convs[-1].bias, final_bias)


class _DecoderBlock(nn.Module):
    """Bilinear x2 upsample, then a residual that starts at zero.

    `out = up(x) + conv2(relu(gn(conv1(up(x)))))` with conv2 zero-initialised,
    so the block is exactly bilinear upsampling until training moves it. See
    the comment at its construction site for why that matters here.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        _kaiming_init_conv(self.conv1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return x + self.conv2(F.relu(self.gn(self.conv1(x)), inplace=True))


class DirectorCenterNet(nn.Module):
    """Director-CenterNet: Heatmap-based Multi-Region Viewport Prediction (MRVP).

    Features:
    - ResNet-50 + FPN backbone
    - Ranked multi-peak heatmap head, offset head, size head
    - Training objectives: L_hcm, L_rmc, L_rep, L_smooth, L_off, L_size
    - Ranked multi-peak extraction at inference with threshold tau=0.2 and K=3
    """

    def __init__(
        self,
        in_channels: int = 36,
        num_classes: int = 1,
        down_ratio: int = 4,
        head_conv: int = config.DIRECTOR_HEAD_CONV,
        k_max: int = config.DIRECTOR_K,
        conf_threshold: float = config.DIRECTOR_TAU,
        render_sigma: float = config.DIRECTOR_RENDER_SIGMA,
        u_observers: int = config.NUM_OBSERVERS_U,
        viewport_size_hw: Tuple[int, int] = config.VIEWPORT_SIZE_HW,
        loss_weights: Optional[Dict[str, float]] = None,
        smooth_huber_delta: float = config.DIRECTOR_SMOOTH_HUBER_DELTA,
        smooth_warmup_start: int = config.DIRECTOR_SMOOTH_WARMUP_START,
        smooth_warmup_full: int = config.DIRECTOR_SMOOTH_WARMUP_FULL,
        soft_center_radius: int = config.DIRECTOR_SOFT_CENTER_RADIUS,
        peak_border_margin: int = config.DIRECTOR_PEAK_BORDER_MARGIN,
        trainable_layers: int = config.DIRECTOR_TRAINABLE_LAYERS,
        dense_positives: bool = config.DIRECTOR_DENSE_POSITIVES,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.down_ratio = int(down_ratio)
        if self.down_ratio > FPN_STRIDE or FPN_STRIDE % self.down_ratio != 0:
            raise ValueError(
                f"down_ratio must be a power-of-two divisor of the FPN level this "
                f"model reads (stride {FPN_STRIDE}); got {down_ratio}."
            )
        self.k_max = k_max
        self.conf_threshold = conf_threshold
        self.render_sigma = render_sigma
        self.u_observers = u_observers
        self.viewport_size_hw = viewport_size_hw
        self.smooth_huber_delta = float(smooth_huber_delta)
        self.smooth_warmup_start = int(smooth_warmup_start)
        self.smooth_warmup_full = int(smooth_warmup_full)
        self.soft_center_radius = int(soft_center_radius)
        self.peak_border_margin = int(peak_border_margin)
        self.trainable_layers = int(trainable_layers)
        self.dense_positives = bool(dense_positives)
        self.head_conv = int(head_conv)

        # `soft_center_radius`, `peak_border_margin` and the NMS kernel are
        # configured in cells but mean a distance in tiles, so each is
        # re-derived from the output stride. Keeping the configured values on
        # self untouched matters: _write_run_provenance records them and
        # inference rebuilds the model from that record, so scaling them in
        # place would apply the factor a second time on every reload.
        cells_per_tile = 1.0 / self.down_ratio
        self._soft_center_cells = max(
            1, int(round(soft_center_radius * FPN_STRIDE * cells_per_tile))
        )
        self._peak_border_cells = max(
            0, int(round(peak_border_margin * FPN_STRIDE * cells_per_tile))
        )
        self._nms_kernel = 1 + 2 * max(
            1, int(round(NMS_RADIUS_TILES * cells_per_tile))
        )
        # The focal negative term sums over every cell while num_pos is fixed
        # at one per sample, so its magnitude would grow with the square of the
        # resolution and, under a single global gradient-norm clip, starve
        # L_rmc, L_rep and L_smooth exactly as the old squared-L2 L_smooth did.
        # Weighting each cell by the map area it covers makes the sum an
        # approximation of the same integral at any stride; it is 1.0 at the
        # stride the loss weights were calibrated at.
        self._hcm_neg_weight = (self.down_ratio / FPN_STRIDE) ** 2
        # Updated per epoch by train.py; only L_smooth's warmup reads it.
        self._current_epoch = self.smooth_warmup_full

        default_weights = {
            "lambda_hcm": 1.0,
            "lambda_off": 1.0,
            "lambda_sz": 0.1,
            "lambda_rmc": 0.5,
            "lambda_rep": 0.3,
            # 0.2 with a squared-L2 L_smooth collapsed every ablation that
            # carried the term; see trajectory_smoothness_loss. 0.02 was the
            # first safe value after the reshaping and turned out to be inert -
            # full and no_smooth agreed to four decimals on every metric. A
            # sweep at 0.1 improved both VD (3.46 -> 3.30) and IR
            # (0.592 -> 0.601), so that is the calibrated value.
            "lambda_sm": 0.1,
        }
        if loss_weights is not None:
            default_weights.update(loss_weights)
        self.loss_weights = default_weights

        # Backbone: ResNet-50 with FPN.
        #
        # trainable_layers defaults to 3 in torchvision, which freezes layer1 -
        # and layer1 is exactly the level this model reads, since the heads sit
        # on fpn['0']. Mask R-CNN shares that default but routes around it: its
        # ROI head pools several FPN levels and scores them with a trainable
        # MLP, so a frozen layer1 costs it much less. Here the only feature map
        # the heatmap ever sees would be ImageNet weights applied to StarCraft
        # tile tensors, which are not natural images. We unfreeze it.
        try:
            self.backbone = resnet_fpn_backbone(
                "resnet50", weights="DEFAULT", trainable_layers=trainable_layers
            )
        except Exception as e:
            Logger.warn(f"[DirectorCenterNet] Failed to load ImageNet weights ({e}), fallback to uninitialized backbone.")
            self.backbone = resnet_fpn_backbone(
                "resnet50", weights=None, trainable_layers=trainable_layers
            )

        self.backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        _kaiming_init_conv(self.backbone.body.conv1)

        # Stride 4 feature level from FPN output '0' (channels = 256)
        fpn_out_channels = 256

        # Optional upsampling decoder, when the heads are asked for an output
        # stride finer than the FPN level.
        #
        # Stride 4 is disproportionately coarse for this target: the viewport
        # is 12x20 tiles, i.e. 3x5 cells, where CenterNet runs stride 4 on
        # objects spanning tens of cells. Sub-cell precision is not the limit
        # (the offset head decodes position continuously); the density of
        # candidate peaks is, since two ranked modes 12 tiles apart are only 3
        # cells apart at stride 4. Upsampling here rather than restriding
        # conv1 keeps the ImageNet-pretrained backbone and costs ~16x less.
        # Each block is bilinear upsampling followed by a residual whose last
        # conv is zero-initialised, so at step 0 the decoder is *exactly*
        # bilinear upsampling and the heads see the same pretrained FPN
        # semantics they see at stride 4.
        #
        # The first version of this stacked plain conv-norm-ReLU blocks with
        # Kaiming init, which destroys those semantics at step 0: the heads
        # then have to learn from noise while the decoder learns to pass
        # information at all. Under the old learning-rate schedule that had six
        # usable epochs the stride-2 run never recovered - 66% frame coverage,
        # median peak score 0.275 against 0.572 at stride 4, which is a model
        # that never became confident rather than one that ran out of capacity.
        # Starting from identity means a stride change tests resolution instead
        # of testing whether a fresh decoder can be optimised.
        n_up = int(round(math.log2(FPN_STRIDE / self.down_ratio)))
        self.decoder = nn.Sequential(
            *[_DecoderBlock(fpn_out_channels) for _ in range(n_up)]
        ) if n_up else nn.Identity()

        # Heatmap Head (2x 3x3 conv + 1x1 conv)
        self.hm_head = nn.Sequential(
            nn.Conv2d(fpn_out_channels, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_classes, kernel_size=1),
        )
        # Offset Head (predicts sub-pixel offsets in feature coordinates)
        self.off_head = nn.Sequential(
            nn.Conv2d(fpn_out_channels, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1),
        )
        # Size Head (predicts width, height in tile units / down_ratio)
        self.wh_head = nn.Sequential(
            nn.Conv2d(fpn_out_channels, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1),
        )

        # Initialize heads. -2.19 puts the initial heatmap probability at ~0.1,
        # which only holds now that the output conv no longer overwhelms it.
        _init_head(self.hm_head, final_bias=-2.19)
        _init_head(self.off_head)
        _init_head(self.wh_head)
        # The viewport is a fixed-size camera rectangle, so the size head's
        # answer is known up front. Starting its bias there (in feature units)
        # means it begins at the target instead of at 0, where the min=2.0
        # clamp would otherwise hold every predicted region at 2x2 tiles until
        # L_size grows it - and L_rep sees no overlap to penalise meanwhile.
        with torch.no_grad():
            self.wh_head[-1].bias.copy_(
                torch.tensor(
                    [viewport_size_hw[1] / down_ratio, viewport_size_hw[0] / down_ratio],
                    dtype=self.wh_head[-1].bias.dtype,
                )
            )

    def set_epoch(self, epoch: int) -> None:
        """Tell the model which epoch it is, for L_smooth's warmup ramp.

        train_model() calls this on every module that exposes it, so it works
        through KBRSWrapper and any other wrapper without them forwarding it.
        """
        self._current_epoch = int(epoch)

    def smooth_warmup_scale(self) -> float:
        """0 before `smooth_warmup_start`, linear to 1 at `smooth_warmup_full`.

        L_smooth is a statement about the trajectory of a camera that is
        tracking something. Applying it from step 0 asks that of a randomly
        initialised heatmap, and the cheapest way to satisfy it is to stop
        moving - which is the solution the collapsed ablations found. Holding
        it off until the primary region is roughly right makes it a
        regulariser on a real trajectory instead of a shortcut.
        """
        epoch = self._current_epoch
        if epoch < self.smooth_warmup_start:
            return 0.0
        if epoch >= self.smooth_warmup_full:
            return 1.0
        span = max(1, self.smooth_warmup_full - self.smooth_warmup_start)
        return float(epoch - self.smooth_warmup_start) / float(span)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        fpn_feats = self.backbone(x)
        # '0' corresponds to stride 4; the decoder is Identity unless the
        # configured output stride is finer than that.
        return self.decoder(fpn_feats["0"])

    def _predict_heads(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # The raw logits are returned alongside the sigmoid map because the
        # soft-argmax centre (see _soft_peak_offsets) needs a softmax over
        # logits; a softmax over already-squashed [0, 1] scores is nearly
        # uniform and would carry almost no positional information.
        hm_logits = self.hm_head(feat)
        pred_hm = torch.sigmoid(hm_logits)
        pred_off = self.off_head(feat)
        pred_wh = self.wh_head(feat)
        return pred_hm, hm_logits, pred_off, pred_wh

    def _soft_peak_offsets(
        self,
        hm_logits_single: torch.Tensor,
        sel_ys: torch.Tensor,
        sel_xs: torch.Tensor,
        radius: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expected sub-cell displacement of the heatmap mass around each peak.

        `torch.topk` hands back *indices*, which carry no gradient. A centre
        built as (index + offset_head) is therefore a function of the offset
        head alone: L_smooth and L_rep can reach `off_head` and `wh_head` but
        never `hm_head`. The probe in scripts/probe_director.py measures
        exactly that - L_smooth puts a gradient norm of 6.5e4 on the offset
        head and precisely 0 on the heatmap head.

        That is why scaling L_smooth down is necessary but not sufficient: the
        objective is meant to stabilise *where the model looks*, and it has no
        path to the map that decides it. Instead it dumps an unsatisfiable
        gradient (the offset spans one cell; the displacement spans tens of
        tiles) into the offset head and, through the shared trunk, corrupts the
        features the heatmap head reads.

        A softmax over the logits in a small window around each peak gives a
        centre that *is* a differentiable function of the heatmap. It returns
        displacements in feature cells, and collapses toward 0 as the peak
        sharpens, so it stays a faithful estimate of the peak position.
        """
        feat_h, feat_w = hm_logits_single.shape
        taps = torch.arange(-radius, radius + 1, device=hm_logits_single.device)
        n = sel_ys.numel()

        win_y = sel_ys.view(n, 1, 1) + taps.view(1, -1, 1)  # (n, W, 1)
        win_x = sel_xs.view(n, 1, 1) + taps.view(1, 1, -1)  # (n, 1, W)
        inside = (
            (win_y >= 0) & (win_y < feat_h) & (win_x >= 0) & (win_x < feat_w)
        ).expand(n, taps.numel(), taps.numel())

        gather_y = win_y.clamp(0, feat_h - 1).expand_as(inside)
        gather_x = win_x.clamp(0, feat_w - 1).expand_as(inside)
        logits = hm_logits_single[gather_y, gather_x]
        # Out-of-map taps must not receive probability mass, otherwise peaks on
        # the border get pulled outward by the padding.
        logits = logits.masked_fill(~inside, float("-inf"))

        weights = torch.softmax(logits.reshape(n, -1), dim=1).view_as(logits)
        d_y = (weights * taps.view(1, -1, 1)).sum(dim=(1, 2))
        d_x = (weights * taps.view(1, 1, -1)).sum(dim=(1, 2))
        return d_y, d_x

    def _extract_predicted_regions_differentiable(
        self,
        pred_hm: torch.Tensor,
        pred_off: torch.Tensor,
        pred_wh: torch.Tensor,
        img_h: int,
        img_w: int,
        conf_thresh: float,
        k_max: int,
        hm_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Extract top-K regions with differentiable box coordinates.

        Two parallel sets of regions come back. `boxes_batch` /
        `primary_centers` are decoded as (peak cell + offset head) and are what
        inference emits. `soft_boxes_batch` / `primary_centers_soft` replace
        the offset head with the soft-argmax displacement of
        _soft_peak_offsets, so they are differentiable with respect to the
        heatmap; L_smooth and L_rep use those. They are kept separate on
        purpose - adding both displacements would double-count the sub-cell
        position and bias the emitted boxes, since only the offset head is
        supervised (by L_off) to represent it.
        """
        b, c, feat_h, feat_w = pred_hm.shape
        stride = self.down_ratio

        # 3x3 max-pooling for NMS. max_pool2d pads with -inf, so a corner cell
        # only has to beat 3 real neighbours and an edge cell 5, against 8 for
        # an interior cell; under a flat heatmap that makes a border cell a
        # local maximum with probability 1/4 rather than 1/9. Border cells were
        # 23% of predictions across every ablation, independent of the loss
        # composition. A mode centre in the outermost cell is also geometrically
        # implausible - it is 0-3 tiles in, where a 12x20 viewport is almost
        # entirely off-map, so the real centre of that camera position is a
        # cell or two further in. Dropping the border ring from the candidates
        # costs nothing reachable and removes the bias.
        hm_max = F.max_pool2d(pred_hm, kernel_size=self._nms_kernel, stride=1, padding=self._nms_kernel // 2)
        keep = (pred_hm == hm_max).float()
        if self._peak_border_cells > 0 and min(feat_h, feat_w) > 2 * self._peak_border_cells:
            m = self._peak_border_cells
            border = torch.zeros_like(keep)
            border[..., m:feat_h - m, m:feat_w - m] = 1.0
            keep = keep * border
        scored_hm = pred_hm * keep

        boxes_batch = []
        soft_boxes_batch = []
        scores_batch = []
        labels_batch = []
        primary_centers = torch.full((b, 2), float("nan"), device=pred_hm.device)
        primary_centers_soft = torch.full((b, 2), float("nan"), device=pred_hm.device)

        default_h, default_w = self.viewport_size_hw

        for i in range(b):
            flat_scores = scored_hm[i, 0].view(-1)
            num_peaks = min(k_max, flat_scores.size(0))
            topk_scores, topk_inds = torch.topk(flat_scores, num_peaks)

            # Filter by threshold tau (variable number of outputs)
            mask = topk_scores >= conf_thresh
            if not mask.any():
                # If no peak exceeds threshold, keep at least the top-1 peak for primary continuity
                mask = torch.zeros_like(mask)
                mask[0] = True

            sel_scores = topk_scores[mask]
            sel_inds = topk_inds[mask]

            sel_ys = (sel_inds // feat_w).float()
            sel_xs = (sel_inds % feat_w).float()

            off_x = pred_off[i, 0].view(-1)[sel_inds]
            off_y = pred_off[i, 1].view(-1)[sel_inds]

            # Differentiable center positions
            cx = (sel_xs + off_x) * stride
            cy = (sel_ys + off_y) * stride

            wh_w = pred_wh[i, 0].view(-1)[sel_inds] * stride
            wh_h = pred_wh[i, 1].view(-1)[sel_inds] * stride
            # Clamp sizes to prevent collapse
            w = torch.clamp(wh_w, min=2.0, max=float(img_w))
            h = torch.clamp(wh_h, min=2.0, max=float(img_h))

            # Primary region center
            primary_centers[i, 0] = cx[0]
            primary_centers[i, 1] = cy[0]

            if hm_logits is not None:
                d_y, d_x = self._soft_peak_offsets(
                    hm_logits[i, 0],
                    torch.div(sel_inds, feat_w, rounding_mode="floor"),
                    sel_inds % feat_w,
                    self._soft_center_cells,
                )
                soft_cx = (sel_xs + d_x) * stride
                soft_cy = (sel_ys + d_y) * stride
                primary_centers_soft[i, 0] = soft_cx[0]
                primary_centers_soft[i, 1] = soft_cy[0]

            # Shift the region inside the map instead of cropping it, matching
            # metrics.modes.box_from_center, which is what evaluation applies.
            # Cropping returned a narrower box at the edges, so L_rep saw
            # regions that were not the fixed-size viewport the task is defined
            # over and could reduce an IoU penalty by sliding a region off the
            # map. `predictions_from_dets(size_wh=...)` already re-imposes the
            # true size downstream; this makes the model agree with it.
            x1 = torch.clamp(cx - w / 2.0, torch.zeros_like(w), (float(img_w) - w).clamp_min(0.0))
            y1 = torch.clamp(cy - h / 2.0, torch.zeros_like(h), (float(img_h) - h).clamp_min(0.0))
            x2 = x1 + w
            y2 = y1 + h

            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            labels = torch.ones_like(sel_scores, dtype=torch.int64)

            if hm_logits is not None:
                sx1 = torch.clamp(soft_cx - w / 2.0, torch.zeros_like(w), (float(img_w) - w).clamp_min(0.0))
                sy1 = torch.clamp(soft_cy - h / 2.0, torch.zeros_like(h), (float(img_h) - h).clamp_min(0.0))
                soft_boxes_batch.append(torch.stack([sx1, sy1, sx1 + w, sy1 + h], dim=1))
            else:
                soft_boxes_batch.append(boxes)

            boxes_batch.append(boxes)
            scores_batch.append(sel_scores)
            labels_batch.append(labels)

        return (
            boxes_batch,
            scores_batch,
            labels_batch,
            primary_centers,
            primary_centers_soft,
            soft_boxes_batch,
        )

    def forward(
        self,
        images: Union[List[torch.Tensor], torch.Tensor],
        targets: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        batched = torch.stack(images, dim=0) if isinstance(images, list) else images
        device = batched.device
        b_size, _, img_h, img_w = batched.shape

        if self.training:
            if targets is None:
                raise ValueError("Targets must be provided during training.")

            # L_smooth only needs the next frame's image; 'next_modes' rides on
            # the mode cache and can be absent even when pairing is active.
            has_pairs = any("next_image" in t for t in targets)

            feat = self._extract_features(batched)
            pred_hm, hm_logits, pred_off, pred_wh = self._predict_heads(feat)
            _, _, feat_h, feat_w = pred_hm.shape
            stride = self.down_ratio

            # Render Gaussian targets (Eq 12). A sample whose mode cache entry
            # is missing yields an empty dict here and is skipped by the
            # renderer (no Top-1 target, no offset/size target for it).
            curr_modes = [t.get("modes", {}) for t in targets]
            obs_boxes = (
                [t.get("boxes") for t in targets] if self.dense_positives else None
            )
            rendered = render_gaussian_heatmap_targets(
                modes_list=curr_modes,
                batch_size=b_size,
                feat_h=feat_h,
                feat_w=feat_w,
                stride=stride,
                device=device,
                render_sigma=self.render_sigma,
                u_observers=self.u_observers,
                viewport_size_hw=self.viewport_size_hw,
                obs_boxes_list=obs_boxes,
            )

            # 1. L_hcm (Human Consensus Match loss)
            # The joint target, not Y1, weights the negative term: Y1 is ~0 at
            # the auxiliary modes, so weighting by it made this loss erase
            # exactly what L_rmc creates. The auxiliary support is then left
            # out of the negative domain entirely and handed to L_rmc.
            #
            # That hand-off is conditional on L_rmc being enabled, and the
            # condition is not cosmetic. Applied unconditionally it meant
            # lambda_rmc=0 removed the positive pull on the auxiliary modes
            # while still protecting them from suppression, so the "no L_rmc"
            # ablation was not an ablation of the auxiliary mechanism at all -
            # hcm_only still emitted 2.75 regions per frame. With nothing
            # supervising those cells, leaving them out of the negative domain
            # only makes them unconstrained, which is not a defensible default
            # either. So when L_rmc is off, L_hcm owns the whole map.
            aux_ignore = (
                rendered["aux_support"] if self.loss_weights["lambda_rmc"] > 0.0 else None
            )
            loss_hcm = human_consensus_match_loss(
                pred_hm=pred_hm,
                target_hm_top1=rendered["Y_all"],
                pos_mask=rendered["pos_mask_top1"],
                ignore_mask=aux_ignore,
                neg_weight=self._hcm_neg_weight,
            )

            # 2. L_rmc (Ranked Mode Coverage loss)
            loss_rmc = ranked_mode_coverage_loss(
                pred_hm=pred_hm,
                target_hm_minus=rendered["Y_minus"],
                mask_omega=rendered["mask_omega"],
                has_aux=rendered["has_aux"],
                aux_support=rendered["aux_support"],
            )

            # 3. Extract predicted regions for L_rep and L_smooth
            (
                pred_boxes,
                pred_scores,
                _,
                primary_centers,
                primary_centers_soft,
                pred_boxes_soft,
            ) = self._extract_predicted_regions_differentiable(
                pred_hm=pred_hm,
                pred_off=pred_off,
                pred_wh=pred_wh,
                img_h=img_h,
                img_w=img_w,
                conf_thresh=self.conf_threshold,
                k_max=self.k_max,
                hm_logits=hm_logits,
            )

            # 4. L_rep (Spatial Repulsion loss). Soft boxes for the same reason
            # L_smooth uses soft centres: boxes decoded from topk indices are a
            # function of the offset and size heads only, so the penalty could
            # never push the *peaks* apart - only shrink or nudge the regions
            # around them. That is why adding L_rep moved almost nothing
            # (hcm_rmc IR 0.633 -> no_smooth 0.630).
            loss_rep = spatial_repulsion_loss(pred_boxes_soft)

            # 5. L_smooth (Trajectory Smoothness loss).
            #
            # The next frame needs a second full forward pass through the
            # backbone, which roughly doubles activation memory and step time.
            # When the effective weight is zero - every lambda_sm=0 ablation,
            # and every run during the warmup epochs - that pass is computed
            # and then multiplied by zero, so skipping it is numerically
            # identical (there is no dropout and the norms are frozen, so no
            # RNG is consumed either) and frees about 40% of the memory.
            smooth_weight = self.loss_weights["lambda_sm"] * self.smooth_warmup_scale()
            if has_pairs and smooth_weight > 0.0:
                # Next frame images from paired targets
                next_images = [t["next_image"].to(device) for t in targets if "next_image" in t]
                if len(next_images) == b_size:
                    batched_next = torch.stack(next_images, dim=0)
                    feat_next = self._extract_features(batched_next)
                    (
                        pred_hm_next,
                        hm_logits_next,
                        pred_off_next,
                        pred_wh_next,
                    ) = self._predict_heads(feat_next)
                    (
                        _,
                        _,
                        _,
                        _,
                        primary_centers_next_soft,
                        _,
                    ) = self._extract_predicted_regions_differentiable(
                        pred_hm=pred_hm_next,
                        pred_off=pred_off_next,
                        pred_wh=pred_wh_next,
                        img_h=img_h,
                        img_w=img_w,
                        conf_thresh=self.conf_threshold,
                        k_max=self.k_max,
                        hm_logits=hm_logits_next,
                    )
                    # exclude replay-final windows, which pair with themselves
                    next_valid = torch.tensor(
                        [bool(t.get("next_valid", True)) for t in targets],
                        dtype=torch.bool,
                        device=device,
                    )
                    # Soft centres on both sides: the decoded (cell + offset)
                    # centre is not a function of the heatmap, so using it here
                    # would route this loss into the offset head only.
                    loss_smooth = trajectory_smoothness_loss(
                        primary_centers_soft,
                        primary_centers_next_soft,
                        valid_mask=next_valid,
                        huber_delta=self.smooth_huber_delta,
                    )
                else:
                    loss_smooth = torch.tensor(0.0, device=device)
            else:
                loss_smooth = torch.tensor(0.0, device=device)

            # 6. Offset and Size losses (L1 on valid mode centers)
            all_target_boxes = rendered["target_boxes"]
            all_target_inds = rendered["target_inds"]

            off_losses = []
            sz_losses = []
            vp_h, vp_w = self.viewport_size_hw

            for b in range(b_size):
                b_inds = all_target_inds[b]
                b_boxes = all_target_boxes[b]
                if len(b_inds) == 0:
                    continue

                b_off_pred = pred_off[b].permute(1, 2, 0).view(-1, 2)[b_inds]
                b_wh_pred = pred_wh[b].permute(1, 2, 0).view(-1, 2)[b_inds]

                # GT offset: (cx/stride - floor(cx/stride), cy/stride - floor(cy/stride))
                gt_cx = b_boxes[:, 0]
                gt_cy = b_boxes[:, 1]
                gt_off_x = (gt_cx / stride) - torch.floor(gt_cx / stride)
                gt_off_y = (gt_cy / stride) - torch.floor(gt_cy / stride)
                gt_off = torch.stack([gt_off_x, gt_off_y], dim=1)

                # GT size in feature scale
                gt_wh = torch.stack([b_boxes[:, 2] / stride, b_boxes[:, 3] / stride], dim=1)

                off_losses.append(F.l1_loss(b_off_pred, gt_off))
                sz_losses.append(F.l1_loss(b_wh_pred, gt_wh))

            loss_off = torch.stack(off_losses).mean() if off_losses else torch.tensor(0.0, device=device)
            loss_sz = torch.stack(sz_losses).mean() if sz_losses else torch.tensor(0.0, device=device)

            # Joint Objective (Eq 19). detection/engine_safe.py backprops
            # sum(loss_dict.values()), so each entry carries its own lambda and no
            # aggregate is returned: emitting "loss_total" alongside the components
            # added every lambda to itself (lambda_rep=0.0 still weighed 1.0, which
            # silently defeated the leave-one-out ablations).
            return {
                "loss_centernet_hm": self.loss_weights["lambda_hcm"] * loss_hcm,
                "loss_rmc": self.loss_weights["lambda_rmc"] * loss_rmc,
                "loss_rep": self.loss_weights["lambda_rep"] * loss_rep,
                "loss_smooth": smooth_weight * loss_smooth,
                "loss_off": self.loss_weights["lambda_off"] * loss_off,
                "loss_wh": self.loss_weights["lambda_sz"] * loss_sz,
            }

        else:
            # Inference mode. hm_logits is deliberately not passed: the soft
            # centre exists only to carry gradient, and the emitted boxes stay
            # the decoded (peak cell + offset head) ones, unchanged by this
            # revision.
            feat = self._extract_features(batched)
            pred_hm, _hm_logits, pred_off, pred_wh = self._predict_heads(feat)

            pred_boxes, pred_scores, pred_labels, _, _, _ = self._extract_predicted_regions_differentiable(
                pred_hm=pred_hm,
                pred_off=pred_off,
                pred_wh=pred_wh,
                img_h=img_h,
                img_w=img_w,
                conf_thresh=self.conf_threshold,
                k_max=self.k_max,
            )

            results = []
            for b in range(b_size):
                results.append({
                    "boxes": pred_boxes[b].detach(),
                    "scores": pred_scores[b].detach(),
                    "labels": pred_labels[b].detach(),
                })
            return results
