# src/model/backbones/deformable_detr.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from ..utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class SpatiotemporalDeformableAttention(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 3)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, memory: torch.Tensor, ref_points: torch.Tensor) -> torch.Tensor:
        B, N_q, _ = query.shape
        _, T, H, W, _ = memory.shape

        value = self.value_proj(memory).view(B, T, H, W, self.n_heads, self.head_dim)

        offsets = self.sampling_offsets(query).view(B, N_q, self.n_heads, self.n_points, 3)
        offsets = torch.tanh(offsets)

        attn_weights = self.attention_weights(query).view(B, N_q, self.n_heads, self.n_points)
        attn_weights = F.softmax(attn_weights, dim=-1)

        ref_coords = ref_points.unsqueeze(2).unsqueeze(3) * 2.0 - 1.0
        sample_locations = ref_coords + offsets * 0.2

        sampled_features = []
        for h_idx in range(self.n_heads):
            v_head = value[:, :, :, :, h_idx, :].permute(0, 4, 1, 2, 3)
            grid = sample_locations[:, :, h_idx, :, :].reshape(B, N_q * self.n_points, 1, 1, 3)

            sampled = F.grid_sample(v_head, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            sampled = sampled.view(B, self.head_dim, N_q, self.n_points).permute(0, 2, 3, 1)

            weights = attn_weights[:, :, h_idx, :].unsqueeze(-1)
            head_out = (sampled * weights).sum(dim=2)
            sampled_features.append(head_out)

        out = torch.cat(sampled_features, dim=-1)
        return self.output_proj(out)

class DeformableDETRBackbone(nn.Module):
    """
    Pure Deformable DETR Backbone architecture with Spatiotemporal Deformable Attention.
    """
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        window_size: int = 4,
        num_queries: int = 100,
        d_model: int = 256,
        n_heads: int = 8,
        num_decoder_layers: int = 4,
        use_probabilistic_query: bool = False,
        loss_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.window_size = window_size
        self.num_queries = num_queries
        self.d_model = d_model
        self.use_probabilistic_query = use_probabilistic_query

        default_weights = {
            "loss_ce": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_kl": 0.1
        }
        if loss_weights is not None:
            default_weights.update(loss_weights)
        self.loss_weights = default_weights

        self.per_frame_channels = max(1, in_channels // window_size)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.per_frame_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )

        if self.use_probabilistic_query:
            from ..plugins.probabilistic_query import ProbabilisticLatentQuery
            self.latent_query_gen = ProbabilisticLatentQuery(num_queries=num_queries, d_model=d_model, latent_dim=64)
        else:
            self.query_embed = nn.Embedding(num_queries, d_model)

        self.decoder_layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                "cross_attn": SpatiotemporalDeformableAttention(d_model, n_heads, n_points=4),
                "norm1": nn.LayerNorm(d_model),
                "norm2": nn.LayerNorm(d_model),
                "norm3": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(d_model * 4, d_model)
                )
            }) for _ in range(num_decoder_layers)
        ])

        self.ref_point_head = nn.Linear(d_model, 3)
        self.class_head = nn.Linear(d_model, num_classes)
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4)
        )

    def _extract_spatiotemporal_features(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        T = self.window_size

        if C != self.per_frame_channels * T:
            frames = [x] * T
        else:
            frames = torch.chunk(x, chunks=T, dim=1)

        memories = []
        for t_frame in frames:
            feat_t = self.feature_extractor(t_frame)
            feat_t = feat_t.permute(0, 2, 3, 1)
            memories.append(feat_t)

        return torch.stack(memories, dim=1)

    def _hungarian_matching(self, pred_logits: torch.Tensor, pred_boxes: torch.Tensor, targets: List[Dict[str, torch.Tensor]]):
        indices = []
        for b, target in enumerate(targets):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]

            if len(gt_boxes) == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue

            img_h, img_w = target.get("image_size", (600, 800))
            gt_cxcywh = torch.zeros_like(gt_boxes)
            gt_cxcywh[:, 0] = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0 / img_w
            gt_cxcywh[:, 1] = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0 / img_h
            gt_cxcywh[:, 2] = (gt_boxes[:, 2] - gt_boxes[:, 0]) / img_w
            gt_cxcywh[:, 3] = (gt_boxes[:, 3] - gt_boxes[:, 1]) / img_h

            b_pred_logits = pred_logits[b]
            b_pred_boxes = pred_boxes[b]

            cost_class = -b_pred_logits.softmax(dim=-1)[:, (gt_labels - 1).long()]
            cost_bbox = torch.cdist(b_pred_boxes, gt_cxcywh, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(b_pred_boxes), gt_boxes / torch.tensor([img_w, img_h, img_w, img_h], device=gt_boxes.device))

            cost_matrix = cost_class + 2.0 * cost_bbox + 2.0 * cost_giou

            matched_src, matched_tgt = [], []
            available_src = list(range(self.num_queries))
            for tgt_idx in range(len(gt_boxes)):
                if not available_src:
                    break
                best_src = min(available_src, key=lambda s: cost_matrix[s, tgt_idx].item())
                matched_src.append(best_src)
                matched_tgt.append(tgt_idx)
                available_src.remove(best_src)

            indices.append((
                torch.tensor(matched_src, dtype=torch.long, device=pred_logits.device),
                torch.tensor(matched_tgt, dtype=torch.long, device=pred_logits.device)
            ))

        return indices

    def forward(self, images, targets=None):
        batched = torch.stack(images, dim=0) if isinstance(images, list) else images
        B, _, H, W = batched.shape

        memory = self._extract_spatiotemporal_features(batched)
        if self.use_probabilistic_query:
            queries, kl_loss = self.latent_query_gen(batch_size=B, is_training=self.training)
        else:
            queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
            kl_loss = torch.tensor(0.0, device=batched.device)

        ref_points = torch.sigmoid(self.ref_point_head(queries))

        output = queries
        for layer in self.decoder_layers:
            sa_out, _ = layer["self_attn"](output, output, output)
            output = layer["norm1"](output + sa_out)

            ca_out = layer["cross_attn"](output, memory, ref_points)
            output = layer["norm2"](output + ca_out)

            ffn_out = layer["ffn"](output)
            output = layer["norm3"](output + ffn_out)

        pred_logits = self.class_head(output)
        pred_boxes = torch.sigmoid(self.box_head(output))

        if self.training:
            if targets is None:
                raise ValueError("Targets must be passed during training mode")

            indices = self._hungarian_matching(pred_logits, pred_boxes, targets)

            loss_ce = torch.tensor(0.0, device=batched.device)
            loss_bbox = torch.tensor(0.0, device=batched.device)
            loss_giou = torch.tensor(0.0, device=batched.device)
            num_matches = 0

            for b, (src_idx, tgt_idx) in enumerate(indices):
                if len(src_idx) == 0:
                    continue

                target = targets[b]
                gt_labels = (target["labels"][tgt_idx] - 1).long()
                gt_boxes = target["boxes"][tgt_idx]

                gt_cxcywh = torch.zeros_like(gt_boxes)
                gt_cxcywh[:, 0] = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0 / W
                gt_cxcywh[:, 1] = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0 / H
                gt_cxcywh[:, 2] = (gt_boxes[:, 2] - gt_boxes[:, 0]) / W
                gt_cxcywh[:, 3] = (gt_boxes[:, 3] - gt_boxes[:, 1]) / H

                loss_ce += F.cross_entropy(pred_logits[b, src_idx], gt_labels)
                loss_bbox += F.l1_loss(pred_boxes[b, src_idx], gt_cxcywh)

                pred_xyxy = box_cxcywh_to_xyxy(pred_boxes[b, src_idx])
                gt_xyxy = gt_boxes / torch.tensor([W, H, W, H], device=gt_boxes.device)
                giou_matrix = generalized_box_iou(pred_xyxy, gt_xyxy)
                loss_giou += (1.0 - torch.diag(giou_matrix)).mean()

                num_matches += 1

            if num_matches > 0:
                loss_ce /= num_matches
                loss_bbox /= num_matches
                loss_giou /= num_matches

            total_loss = (
                self.loss_weights["loss_ce"] * loss_ce +
                self.loss_weights["loss_bbox"] * loss_bbox +
                self.loss_weights["loss_giou"] * loss_giou +
                self.loss_weights["loss_kl"] * kl_loss
            )

            return {
                "loss_ce": loss_ce,
                "loss_bbox": loss_bbox,
                "loss_giou": loss_giou,
                "loss_kl": kl_loss,
                "loss_total": total_loss
            }
        else:
            results = []
            prob = pred_logits.softmax(dim=-1)

            for b in range(B):
                # Class 0: Foreground unit (Viewport), Class 1: Background
                # Extract true foreground score (prob[:, 0]) instead of max over background
                scores = prob[b, :, 0]
                labels = torch.ones_like(scores, dtype=torch.long)

                b_boxes = pred_boxes[b]

                cx, cy, w, h = b_boxes.unbind(-1)
                x1 = (cx - w / 2.0) * W
                y1 = (cy - h / 2.0) * H
                x2 = (cx + w / 2.0) * W
                y2 = (cy + h / 2.0) * H

                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

                results.append({
                    "boxes": boxes_xyxy,
                    "scores": scores,
                    "labels": labels
                })

            return results
