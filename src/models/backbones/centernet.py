# src/model/backbones/centernet.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

def focal_loss_keypoints(pred: torch.Tensor, gt: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    pos_mask = gt.eq(1.0).float()
    neg_mask = gt.lt(1.0).float()

    pos_loss = torch.log(pred.clamp_min(1e-6)) * torch.pow(1.0 - pred, alpha) * pos_mask
    neg_loss = torch.log((1.0 - pred).clamp_min(1e-6)) * torch.pow(pred, alpha) * torch.pow(1.0 - gt, beta) * neg_mask

    num_pos = pos_mask.sum().clamp_min(1.0)
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    return -(pos_loss + neg_loss) / num_pos

class CenterNetBackbone(nn.Module):
    """
    Pure CenterNet Keypoint-based Object Detection Backbone.
    Predicts Heatmaps, Width/Height, and Offsets.
    """
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 36,
        down_ratio: int = 4,
        max_objs: int = 100,
        head_conv: int = 64,
        loss_weights: Optional[Dict[str, float]] = None,
        use_density_peak: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.down_ratio = down_ratio
        self.max_objs = max_objs
        self.use_density_peak = use_density_peak

        default_weights = {
            "loss_hm": 1.0,
            "loss_wh": 0.1,
            "loss_off": 1.0,
            "loss_density": 0.5,
            "loss_delta": 0.5
        }
        if loss_weights is not None:
            default_weights.update(loss_weights)
        self.loss_weights = default_weights

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.hm_head = nn.Sequential(
            nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_classes, kernel_size=1)
        )
        self.wh_head = nn.Sequential(
            nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1)
        )
        self.off_head = nn.Sequential(
            nn.Conv2d(256, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1)
        )

        # CenterNet heatmap prior: sigmoid(-2.19) ~= 0.1, so training does not start
        # with every cell claiming p=0.5 against ~1000 negatives per positive.
        self.hm_head[-1].bias.data.fill_(-2.19)

        if self.use_density_peak:
            from ..plugins.density_peak import DensityPeakHead
            self.density_peak_head = DensityPeakHead(in_channels=256, head_conv=head_conv)

    def _generate_targets(self, targets: List[Dict], batch_size: int, feat_h: int, feat_w: int, device: torch.device):
        gt_hm = torch.zeros((batch_size, self.num_classes, feat_h, feat_w), device=device)
        gt_wh = torch.zeros((batch_size, self.max_objs, 2), device=device)
        gt_off = torch.zeros((batch_size, self.max_objs, 2), device=device)
        gt_mask = torch.zeros((batch_size, self.max_objs), device=device)
        gt_ind = torch.zeros((batch_size, self.max_objs), dtype=torch.long, device=device)
        gt_density = torch.zeros((batch_size, 1, feat_h, feat_w), device=device)
        gt_delta = torch.zeros((batch_size, 1, feat_h, feat_w), device=device)

        y_grid, x_grid = torch.meshgrid(
            torch.arange(feat_h, device=device),
            torch.arange(feat_w, device=device),
            indexing="ij"
        )

        for b, target in enumerate(targets):
            boxes = target["boxes"]
            labels = target["labels"]

            density_accum = torch.zeros((feat_h, feat_w), device=device)

            for k, (box, label) in enumerate(zip(boxes, labels)):
                if k >= self.max_objs:
                    break

                cls_id = int(label.item()) - 1
                if cls_id < 0 or cls_id >= self.num_classes:
                    continue

                x1, y1, x2, y2 = box.tolist()
                w = (x2 - x1) / self.down_ratio
                h = (y2 - y1) / self.down_ratio
                cx = (x1 + x2) / 2.0 / self.down_ratio
                cy = (y1 + y2) / 2.0 / self.down_ratio

                ct_x = int(cx)
                ct_y = int(cy)

                if ct_x < 0 or ct_x >= feat_w or ct_y < 0 or ct_y >= feat_h:
                    continue

                gt_wh[b, k] = torch.tensor([w, h], device=device)
                gt_off[b, k] = torch.tensor([cx - ct_x, cy - ct_y], device=device)
                gt_mask[b, k] = 1.0
                gt_ind[b, k] = ct_y * feat_w + ct_x

                radius = max(1, int(min(w, h) / 2.0))
                left = max(0, ct_x - 3 * radius)
                right = min(feat_w, ct_x + 3 * radius + 1)
                top = max(0, ct_y - 3 * radius)
                bottom = min(feat_h, ct_y + 3 * radius + 1)

                dist_sq = (x_grid[top:bottom, left:right] - ct_x) ** 2 + (y_grid[top:bottom, left:right] - ct_y) ** 2
                gauss = torch.exp(-dist_sq / (2.0 * (radius ** 2)))

                # Splat the Gaussian into the heatmap target, not just into the
                # density map. Without it gt_hm is a one-hot delta, so the
                # (1 - gt)^beta penalty-reduction term in focal_loss_keypoints is
                # identically 1 and a cell one step off the centre is punished as
                # hard as the far background. maximum() (not +=) keeps overlapping
                # observers from pushing the target above 1.
                gt_hm[b, cls_id, top:bottom, left:right] = torch.maximum(
                    gt_hm[b, cls_id, top:bottom, left:right], gauss
                )
                gt_hm[b, cls_id, ct_y, ct_x] = 1.0  # exact peak stays the positive

                density_accum[top:bottom, left:right] += gauss

            gt_density[b, 0] = density_accum

        return {
            "hm": gt_hm, "wh": gt_wh, "off": gt_off,
            "mask": gt_mask, "ind": gt_ind,
            "density": gt_density, "delta": gt_delta
        }

    def _transpose_and_gather_feat(self, feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), feat.size(2))
        return feat.gather(1, ind)

    def forward(self, images, targets=None):
        batched = torch.stack(images, dim=0) if isinstance(images, list) else images
        device = batched.device

        feat = self.backbone(batched)
        _, _, feat_h, feat_w = feat.shape

        pred_hm = torch.sigmoid(self.hm_head(feat))
        pred_wh = self.wh_head(feat)
        pred_off = self.off_head(feat)

        if self.use_density_peak:
            pred_density, pred_delta = self.density_peak_head(feat)
        else:
            pred_density, pred_delta = None, None

        if self.training:
            if targets is None:
                raise ValueError("Targets must be provided during training")

            gt_dict = self._generate_targets(targets, batched.size(0), feat_h, feat_w, device)

            loss_hm = focal_loss_keypoints(pred_hm, gt_dict["hm"])

            pred_wh_gathered = self._transpose_and_gather_feat(pred_wh, gt_dict["ind"])
            pred_off_gathered = self._transpose_and_gather_feat(pred_off, gt_dict["ind"])
            mask = gt_dict["mask"].unsqueeze(2)

            num_pos = mask.sum().clamp_min(1.0)
            loss_wh = (F.l1_loss(pred_wh_gathered * mask, gt_dict["wh"] * mask, reduction="sum")) / num_pos
            loss_off = (F.l1_loss(pred_off_gathered * mask, gt_dict["off"] * mask, reduction="sum")) / num_pos

            if self.use_density_peak:
                loss_density = F.smooth_l1_loss(pred_density, gt_dict["density"])
                loss_delta = F.smooth_l1_loss(pred_delta, gt_dict["delta"])
            else:
                loss_density = torch.tensor(0.0, device=device)
                loss_delta = torch.tensor(0.0, device=device)

            # detection/engine_safe.py backprops sum(loss_dict.values()), so every
            # entry must already carry its weight and no aggregate may be present.
            # Returning "loss_total" next to its own components counted each one
            # twice: effective weights were (2.0, 1.1, 2.0) instead of (1, 0.1, 1).
            losses = {
                "loss_centernet_hm": self.loss_weights["loss_hm"] * loss_hm,
                "loss_wh": self.loss_weights["loss_wh"] * loss_wh,
                "loss_off": self.loss_weights["loss_off"] * loss_off,
            }
            if self.use_density_peak:
                losses["loss_density"] = self.loss_weights.get("loss_density", 0.5) * loss_density
                losses["loss_delta"] = self.loss_weights.get("loss_delta", 0.5) * loss_delta
            return losses
        else:
            img_h, img_w = batched.shape[-2:]
            results = []

            hm_max = F.max_pool2d(pred_hm, kernel_size=3, stride=1, padding=1)
            keep_mask = (pred_hm == hm_max).float()
            scored_hm = pred_hm * keep_mask

            if self.use_density_peak and pred_density is not None:
                density_factor = torch.sigmoid(pred_density) * (1.0 + torch.tanh(pred_delta))
                final_scores = scored_hm * density_factor
            else:
                final_scores = scored_hm

            for b in range(batched.size(0)):
                b_scores = final_scores[b].view(-1)
                topk_scores, topk_inds = torch.topk(b_scores, min(self.max_objs, b_scores.size(0)))

                topk_clses = (topk_inds // (feat_h * feat_w)).long()
                topk_locs = topk_inds % (feat_h * feat_w)

                topk_ys = (topk_locs // feat_w).float()
                topk_xs = (topk_locs % feat_w).float()

                b_off = pred_off[b].permute(1, 2, 0).view(-1, 2)[topk_locs]
                b_wh = pred_wh[b].permute(1, 2, 0).view(-1, 2)[topk_locs]

                xs = topk_xs + b_off[:, 0]
                ys = topk_ys + b_off[:, 1]
                ws = b_wh[:, 0]
                hs = b_wh[:, 1]

                x1 = (xs - ws / 2.0) * self.down_ratio
                y1 = (ys - hs / 2.0) * self.down_ratio
                x2 = (xs + ws / 2.0) * self.down_ratio
                y2 = (ys + hs / 2.0) * self.down_ratio

                x1 = x1.clamp(0, img_w)
                y1 = y1.clamp(0, img_h)
                x2 = x2.clamp(0, img_w)
                y2 = y2.clamp(0, img_h)

                boxes = torch.stack([x1, y1, x2, y2], dim=1)
                labels = topk_clses + 1

                results.append({
                    "boxes": boxes,
                    "scores": topk_scores,
                    "labels": labels
                })

            return results
