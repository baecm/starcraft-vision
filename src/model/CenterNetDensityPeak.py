# src/model/CenterNetDensityPeak.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

# =====================================================================
# CenterNet Focal Loss for Keypoints
# =====================================================================
def focal_loss_keypoints(pred: torch.Tensor, gt: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """
    Modified Focal Loss for CenterNet keypoint heatmaps.
    pred: (B, C, H, W) after sigmoid in range (0, 1)
    gt: (B, C, H, W) ground truth Gaussian heatmap in range [0, 1]
    """
    pred = torch.clamp(pred, 1e-4, 1.0 - 1e-4)
    pos_inds = gt.eq(1.0).float()
    neg_inds = gt.lt(1.0).float()

    pos_loss = torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos_inds
    neg_loss = torch.log(1.0 - pred) * torch.pow(pred, alpha) * torch.pow(1.0 - gt, beta) * neg_inds

    num_pos = pos_inds.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def gaussian2d(shape: Tuple[int, int], sigma: float = 1.0) -> torch.Tensor:
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = torch.meshgrid(
        torch.arange(-m, m + 1.),
        torch.arange(-n, n + 1.),
        indexing="ij"
    )
    h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap: torch.Tensor, center: Tuple[int, int], radius: int, k: float = 1.0):
    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0], heatmap.shape[1]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right].to(heatmap.device)

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size: Tuple[float, float], min_iou: float = 0.7) -> int:
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_iou) / (1 + min_iou)
    sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_iou) * width * height
    sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_iou
    b3 = -2 * min_iou * (height + width)
    c3 = (min_iou - 1) * width * height
    sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return max(0, int(min(r1, r2, r3)))


# =====================================================================
# Conv Backbone with Upsampling
# =====================================================================
class CenterNetBackbone(nn.Module):
    """
    Backbone for CenterNet accepting arbitrary in_channels.
    Outputs feature map at down_scale = 4.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        # Stem: stride 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        # Stage 1: stride 4 (down_scale=4)
        self.layer1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        # Stage 2: stride 8
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        # Upsampling deconv layer: stride 8 -> stride 4
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.out_channels = base_channels * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.stem(x)       # stride 2
        c2 = self.layer1(c1)    # stride 4
        c3 = self.layer2(c2)    # stride 8
        out = self.upsample(c3) # stride 4
        return out + c2


# =====================================================================
# CenterNet with Density Peak Head Main Class
# =====================================================================
class CenterNetDensityPeak(nn.Module):
    """
    CenterNet model enhanced with Density Peak Head for dense object separation in StarCraft vision.
    """
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        down_ratio: int = 4,
        max_objs: int = 100,
        use_density_peak: bool = True,
        loss_weights: Optional[Dict[str, float]] = None
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

        self.backbone = CenterNetBackbone(in_channels=in_channels, base_channels=64)
        c_in = self.backbone.out_channels

        # 1. Keypoint Heatmap Head
        self.hm_head = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        # Initialize heatmap bias to -2.19 (prior probability ~0.1)
        self.hm_head[-1].bias.data.fill_(-2.19)

        # 2. Width & Height BBox Head
        self.wh_head = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )

        # 3. Sub-pixel Offset Head
        self.off_head = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )

        # 4. Density Peak Head: Local Unit Density (rho) + Cutoff Distance (delta)
        self.density_head = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Softplus()
        )
        self.delta_head = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Softplus()
        )

    def _generate_targets(self, targets: List[Dict[str, torch.Tensor]], batch_size: int, feat_h: int, feat_w: int, device: torch.device):
        gt_hm = torch.zeros((batch_size, self.num_classes, feat_h, feat_w), device=device)
        gt_wh = torch.zeros((batch_size, self.max_objs, 2), device=device)
        gt_off = torch.zeros((batch_size, self.max_objs, 2), device=device)
        gt_ind = torch.zeros((batch_size, self.max_objs), dtype=torch.long, device=device)
        gt_reg_mask = torch.zeros((batch_size, self.max_objs), dtype=torch.float, device=device)

        gt_density = torch.zeros((batch_size, 1, feat_h, feat_w), device=device)
        gt_delta = torch.zeros((batch_size, 1, feat_h, feat_w), device=device)

        for b_idx, target in enumerate(targets):
            boxes = target["boxes"]
            labels = target["labels"]  # 1-indexed

            if len(boxes) == 0:
                continue

            centers = []
            densities = []

            for k_idx, (box, label) in enumerate(zip(boxes, labels)):
                if k_idx >= self.max_objs:
                    break

                cls_id = int(label.item()) - 1
                if cls_id < 0 or cls_id >= self.num_classes:
                    continue

                x1, y1, x2, y2 = box[0].item(), box[1].item(), box[2].item(), box[3].item()
                h, w = (y2 - y1) / self.down_ratio, (x2 - x1) / self.down_ratio
                if h <= 0 or w <= 0:
                    continue

                radius = gaussian_radius((math.ceil(h), math.ceil(w)), min_iou=0.7)
                radius = max(0, int(radius))

                ct = torch.tensor([(x1 + x2) / 2.0 / self.down_ratio, (y1 + y2) / 2.0 / self.down_ratio], device=device)
                ct_int = ct.to(torch.int32)

                if ct_int[0] < 0 or ct_int[0] >= feat_w or ct_int[1] < 0 or ct_int[1] >= feat_h:
                    continue

                draw_umich_gaussian(gt_hm[b_idx, cls_id], (ct_int[0].item(), ct_int[1].item()), radius)

                ind = ct_int[1].item() * feat_w + ct_int[0].item()
                gt_ind[b_idx, k_idx] = ind
                gt_wh[b_idx, k_idx] = torch.tensor([w, h], device=device)
                gt_off[b_idx, k_idx] = ct - ct_int.float()
                gt_reg_mask[b_idx, k_idx] = 1.0

                centers.append(ct)
                densities.append(1.0 + radius * 0.5)

            # Local Density & Delta Peak targets
            if len(centers) > 0:
                centers_t = torch.stack(centers)
                for y in range(feat_h):
                    for x in range(feat_w):
                        pt = torch.tensor([x, y], device=device, dtype=torch.float)
                        dists = torch.norm(centers_t - pt, dim=1)
                        local_density = torch.sum(torch.exp(-dists**2 / (2 * 4.0)))
                        gt_density[b_idx, 0, y, x] = local_density

                        if len(centers) > 1:
                            min_delta = torch.min(dists + 1e-4)
                            gt_delta[b_idx, 0, y, x] = min_delta

        return {
            "hm": gt_hm, "wh": gt_wh, "off": gt_off, "ind": gt_ind,
            "mask": gt_reg_mask, "density": gt_density, "delta": gt_delta
        }

    def _transpose_and_gather_feat(self, feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        return feat

    def forward(self, images, targets=None):
        if isinstance(images, list):
            batched_images = torch.stack(images, dim=0)
        else:
            batched_images = images

        device = batched_images.device
        feat = self.backbone(batched_images)
        _, _, feat_h, feat_w = feat.shape

        pred_hm = torch.sigmoid(self.hm_head(feat))
        pred_wh = self.wh_head(feat)
        pred_off = self.off_head(feat)

        if self.use_density_peak:
            pred_density = self.density_head(feat)
            pred_delta = self.delta_head(feat)
        else:
            pred_density = None
            pred_delta = None

        if self.training:
            if targets is None:
                raise ValueError("Targets must be provided during training")

            gt_dict = self._generate_targets(targets, batched_images.size(0), feat_h, feat_w, device)

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

            total_loss = (
                self.loss_weights["loss_hm"] * loss_hm +
                self.loss_weights["loss_wh"] * loss_wh +
                self.loss_weights["loss_off"] * loss_off +
                (self.loss_weights.get("loss_density", 0.5) * loss_density if self.use_density_peak else 0.0) +
                (self.loss_weights.get("loss_delta", 0.5) * loss_delta if self.use_density_peak else 0.0)
            )

            return {
                "loss_centernet_hm": loss_hm,
                "loss_wh": loss_wh,
                "loss_off": loss_off,
                "loss_density": loss_density,
                "loss_delta": loss_delta,
                "loss_total": total_loss
            }

        else:
            img_h, img_w = batched_images.shape[-2:]
            results = []

            # Density Peak Peak-finding Decoding
            # Apply 3x3 MaxPool NMS on heatmap
            hm_max = F.max_pool2d(pred_hm, kernel_size=3, stride=1, padding=1)
            keep_mask = (pred_hm == hm_max).float()
            scored_hm = pred_hm * keep_mask

            # Density modulation factor
            density_factor = torch.sigmoid(pred_density) * (1.0 + torch.tanh(pred_delta))
            final_scores = scored_hm * density_factor

            for b in range(batched_images.size(0)):
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
                labels = topk_clses + 1  # 1-indexed

                results.append({
                    "boxes": boxes,
                    "scores": topk_scores,
                    "labels": labels
                })

            return results
