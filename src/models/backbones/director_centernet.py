# src/models/backbones/director_centernet.py
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


def _kaiming_init_conv(conv: nn.Conv2d):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


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
        head_conv: int = 64,
        k_max: int = config.DIRECTOR_K,
        conf_threshold: float = config.DIRECTOR_TAU,
        render_sigma: float = config.DIRECTOR_RENDER_SIGMA,
        u_observers: int = config.NUM_OBSERVERS_U,
        viewport_size_hw: Tuple[int, int] = config.VIEWPORT_SIZE_HW,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.down_ratio = down_ratio
        self.k_max = k_max
        self.conf_threshold = conf_threshold
        self.render_sigma = render_sigma
        self.u_observers = u_observers
        self.viewport_size_hw = viewport_size_hw

        default_weights = {
            "lambda_hcm": 1.0,
            "lambda_off": 1.0,
            "lambda_sz": 0.1,
            "lambda_rmc": 0.5,
            "lambda_rep": 0.3,
            "lambda_sm": 0.2,
        }
        if loss_weights is not None:
            default_weights.update(loss_weights)
        self.loss_weights = default_weights

        # Backbone: ResNet-50 with FPN
        try:
            self.backbone = resnet_fpn_backbone("resnet50", weights="DEFAULT")
        except Exception as e:
            Logger.warn(f"[DirectorCenterNet] Failed to load ImageNet weights ({e}), fallback to uninitialized backbone.")
            self.backbone = resnet_fpn_backbone("resnet50", weights=None)

        self.backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        _kaiming_init_conv(self.backbone.body.conv1)

        # Stride 4 feature level from FPN output '0' (channels = 256)
        fpn_out_channels = 256

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

        # Initialize heads
        for m in self.hm_head.modules():
            if isinstance(m, nn.Conv2d):
                _kaiming_init_conv(m)
        for m in self.off_head.modules():
            if isinstance(m, nn.Conv2d):
                _kaiming_init_conv(m)
        for m in self.wh_head.modules():
            if isinstance(m, nn.Conv2d):
                _kaiming_init_conv(m)

        # CenterNet standard bias initialization for heatmap head
        self.hm_head[-1].bias.data.fill_(-2.19)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        fpn_feats = self.backbone(x)
        # '0' corresponds to stride 4
        return fpn_feats["0"]

    def _predict_heads(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_hm = torch.sigmoid(self.hm_head(feat))
        pred_off = self.off_head(feat)
        pred_wh = self.wh_head(feat)
        return pred_hm, pred_off, pred_wh

    def _extract_predicted_regions_differentiable(
        self,
        pred_hm: torch.Tensor,
        pred_off: torch.Tensor,
        pred_wh: torch.Tensor,
        img_h: int,
        img_w: int,
        conf_thresh: float,
        k_max: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Extract top-K regions with differentiable box coordinates."""
        b, c, feat_h, feat_w = pred_hm.shape
        stride = self.down_ratio

        # 3x3 max-pooling for NMS
        hm_max = F.max_pool2d(pred_hm, kernel_size=3, stride=1, padding=1)
        keep = (pred_hm == hm_max).float()
        scored_hm = pred_hm * keep

        boxes_batch = []
        scores_batch = []
        labels_batch = []
        primary_centers = torch.full((b, 2), float("nan"), device=pred_hm.device)

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

            x1 = torch.clamp(cx - w / 2.0, 0.0, float(img_w))
            y1 = torch.clamp(cy - h / 2.0, 0.0, float(img_h))
            x2 = torch.clamp(cx + w / 2.0, 0.0, float(img_w))
            y2 = torch.clamp(cy + h / 2.0, 0.0, float(img_h))

            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            labels = torch.ones_like(sel_scores, dtype=torch.int64)

            boxes_batch.append(boxes)
            scores_batch.append(sel_scores)
            labels_batch.append(labels)

        return boxes_batch, scores_batch, labels_batch, primary_centers

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
            pred_hm, pred_off, pred_wh = self._predict_heads(feat)
            _, _, feat_h, feat_w = pred_hm.shape
            stride = self.down_ratio

            # Render Gaussian targets (Eq 12). A sample whose mode cache entry
            # is missing yields an empty dict here and is skipped by the
            # renderer (no Top-1 target, no offset/size target for it).
            curr_modes = [t.get("modes", {}) for t in targets]
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
            )

            # 1. L_hcm (Human Consensus Match loss)
            loss_hcm = human_consensus_match_loss(
                pred_hm=pred_hm,
                target_hm_top1=rendered["Y1"],
                pos_mask=rendered["pos_mask_top1"],
            )

            # 2. L_rmc (Ranked Mode Coverage loss)
            loss_rmc = ranked_mode_coverage_loss(
                pred_hm=pred_hm,
                target_hm_minus=rendered["Y_minus"],
                mask_omega=rendered["mask_omega"],
                has_aux=rendered["has_aux"],
            )

            # 3. Extract predicted regions for L_rep and L_smooth
            pred_boxes, pred_scores, _, primary_centers = self._extract_predicted_regions_differentiable(
                pred_hm=pred_hm,
                pred_off=pred_off,
                pred_wh=pred_wh,
                img_h=img_h,
                img_w=img_w,
                conf_thresh=self.conf_threshold,
                k_max=self.k_max,
            )

            # 4. L_rep (Spatial Repulsion loss)
            loss_rep = spatial_repulsion_loss(pred_boxes)

            # 5. L_smooth (Trajectory Smoothness loss)
            if has_pairs:
                # Next frame images from paired targets
                next_images = [t["next_image"].to(device) for t in targets if "next_image" in t]
                if len(next_images) == b_size:
                    batched_next = torch.stack(next_images, dim=0)
                    feat_next = self._extract_features(batched_next)
                    pred_hm_next, pred_off_next, pred_wh_next = self._predict_heads(feat_next)
                    _, _, _, primary_centers_next = self._extract_predicted_regions_differentiable(
                        pred_hm=pred_hm_next,
                        pred_off=pred_off_next,
                        pred_wh=pred_wh_next,
                        img_h=img_h,
                        img_w=img_w,
                        conf_thresh=self.conf_threshold,
                        k_max=self.k_max,
                    )
                    # exclude replay-final windows, which pair with themselves
                    next_valid = torch.tensor(
                        [bool(t.get("next_valid", True)) for t in targets],
                        dtype=torch.bool,
                        device=device,
                    )
                    loss_smooth = trajectory_smoothness_loss(
                        primary_centers, primary_centers_next, valid_mask=next_valid
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

            # Joint Objective (Eq 19)
            total_loss = (
                self.loss_weights["lambda_hcm"] * loss_hcm
                + self.loss_weights["lambda_off"] * loss_off
                + self.loss_weights["lambda_sz"] * loss_sz
                + self.loss_weights["lambda_rmc"] * loss_rmc
                + self.loss_weights["lambda_rep"] * loss_rep
                + self.loss_weights["lambda_sm"] * loss_smooth
            )

            return {
                "loss_centernet_hm": loss_hcm,
                "loss_rmc": loss_rmc,
                "loss_rep": loss_rep,
                "loss_smooth": loss_smooth,
                "loss_off": loss_off,
                "loss_wh": loss_sz,
                "loss_total": total_loss,
            }

        else:
            # Inference mode
            feat = self._extract_features(batched)
            pred_hm, pred_off, pred_wh = self._predict_heads(feat)

            pred_boxes, pred_scores, pred_labels, _ = self._extract_predicted_regions_differentiable(
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
