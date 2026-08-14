# src/model/DeformableVideoDETR.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

# =====================================================================
# Bounding Box Utilities & GIoU Computation
# =====================================================================
def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU between two sets of boxes [N, 4] and [M, 4] in (x1, y1, x2, y2) format.
    Returns matrix of shape [N, M].
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)

    lt_c = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, :, 0] * wh_c[:, :, 1]

    giou = iou - (area_c - union) / (area_c + 1e-6)
    return giou

# =====================================================================
# Probabilistic Latent Query Module
# =====================================================================
class ProbabilisticLatentQuery(nn.Module):
    """
    Models Object Queries as latent Gaussian distributions N(mu, sigma^2).
    Allows sampling latent positional/content embeddings during forward pass
    and computes KL-divergence loss for uncertainty modeling in dense battle scenes.
    """
    def __init__(self, num_queries: int = 100, d_model: int = 256, latent_dim: int = 64):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        self.latent_dim = latent_dim

        # Base query embedding
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Probabilistic heads for Gaussian distribution parameters
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.log_var_head = nn.Linear(d_model, latent_dim)

        # Latent to model projection
        self.latent_proj = nn.Linear(latent_dim, d_model)

    def forward(self, batch_size: int, is_training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        base_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1) # (B, Nq, d_model)

        mu = self.mu_head(base_embed)             # (B, Nq, latent_dim)
        log_var = self.log_var_head(base_embed)   # (B, Nq, latent_dim)

        if is_training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        latent_feature = self.latent_proj(z)
        queries = base_embed + latent_feature

        # KL Divergence Loss
        kl_loss = -0.5 * torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()

        return queries, kl_loss

# =====================================================================
# Spatiotemporal Deformable Cross-Attention Module
# =====================================================================
class SpatiotemporalDeformableAttention(nn.Module):
    """
    Multi-Scale Spatiotemporal Deformable Attention across frame windows.
    Samples key features dynamically across spatial (H, W) and temporal (T) dimensions.
    """
    def __init__(self, d_model: int = 256, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 3)  # (dx, dy, dt)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, memory: torch.Tensor, ref_points: torch.Tensor) -> torch.Tensor:
        """
        query: (B, N_q, d_model)
        memory: (B, T, H, W, d_model)
        ref_points: (B, N_q, 3) normalized (x, y, t)
        """
        B, N_q, _ = query.shape
        _, T, H, W, _ = memory.shape

        value = self.value_proj(memory).view(B, T, H, W, self.n_heads, self.head_dim)

        offsets = self.sampling_offsets(query).view(B, N_q, self.n_heads, self.n_points, 3)
        offsets = torch.tanh(offsets)  # Normalized offsets in range (-1, 1)

        attn_weights = self.attention_weights(query).view(B, N_q, self.n_heads, self.n_points)
        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, N_q, n_heads, n_points)

        # Compute sampling locations in normalized coords [-1, 1] for grid_sample
        # ref_points in [0, 1] -> convert to [-1, 1]
        ref_coords = ref_points.unsqueeze(2).unsqueeze(3) * 2.0 - 1.0  # (B, N_q, 1, 1, 3)
        sample_locations = ref_coords + offsets * 0.2  # (B, N_q, n_heads, n_points, 3)

        # Bilinear sampling per head
        sampled_features = []
        for h_idx in range(self.n_heads):
            # Extract single head value: (B, T, H, W, head_dim) -> (B, head_dim, T, H, W)
            v_head = value[:, :, :, :, h_idx, :].permute(0, 4, 1, 2, 3)

            # Sampling grid for this head: (B, N_q * n_points, 1, 1, 3)
            grid = sample_locations[:, :, h_idx, :, :].reshape(B, N_q * self.n_points, 1, 1, 3)

            # 3D Grid sampling: v_head shape (B, C, D, H, W), grid shape (B, D_out, H_out, W_out, 3)
            sampled = F.grid_sample(v_head, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            # sampled shape: (B, head_dim, N_q * n_points, 1, 1)
            sampled = sampled.view(B, self.head_dim, N_q, self.n_points).permute(0, 2, 3, 1) # (B, N_q, n_points, head_dim)

            weights = attn_weights[:, :, h_idx, :].unsqueeze(-1) # (B, N_q, n_points, 1)
            head_out = (sampled * weights).sum(dim=2)            # (B, N_q, head_dim)
            sampled_features.append(head_out)

        out = torch.cat(sampled_features, dim=-1) # (B, N_q, d_model)
        return self.output_proj(out)

# =====================================================================
# Deformable Video DETR Main Model
# =====================================================================
class DeformableVideoDETR(nn.Module):
    """
    Deformable Video DETR with Probabilistic Latent Query.
    Extends Deformable DETR for temporal video frames with uncertainty-aware latent object queries.
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

        # Frame channels per window
        self.per_frame_channels = max(1, in_channels // window_size)

        # Spatiotemporal CNN Feature Extractor
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

        # Latent / Deterministic Query Module
        if self.use_probabilistic_query:
            self.latent_query_gen = ProbabilisticLatentQuery(num_queries=num_queries, d_model=d_model, latent_dim=64)
        else:
            self.query_embed = nn.Embedding(num_queries, d_model)

        # Decoder Blocks
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

        # Reference Points Predictor
        self.ref_point_head = nn.Linear(d_model, 3)

        # Prediction Heads
        self.class_head = nn.Linear(d_model, num_classes)
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4)
        )

    def _extract_spatiotemporal_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, H, W)
        Returns memory tensor of shape (B, T, H', W', d_model)
        """
        B, C, H, W = x.shape
        T = self.window_size

        if C != self.per_frame_channels * T:
            # Fallback if channel count does not divide evenly
            frames = [x] * T
        else:
            frames = torch.chunk(x, chunks=T, dim=1)

        memories = []
        for t_frame in frames:
            feat_t = self.feature_extractor(t_frame) # (B, d_model, H', W')
            feat_t = feat_t.permute(0, 2, 3, 1)       # (B, H', W', d_model)
            memories.append(feat_t)

        memory = torch.stack(memories, dim=1)         # (B, T, H', W', d_model)
        return memory

    def _hungarian_matching(self, pred_logits: torch.Tensor, pred_boxes: torch.Tensor, targets: List[Dict[str, torch.Tensor]]):
        """
        Greedy / Linear sum assignment matching between predicted queries and GT target boxes.
        """
        indices = []
        for b, target in enumerate(targets):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"] # 1-indexed

            if len(gt_boxes) == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue

            # Convert GT boxes (x1, y1, x2, y2) to normalized (cx, cy, w, h)
            img_h, img_w = target.get("image_size", (600, 800))
            gt_cxcywh = torch.zeros_like(gt_boxes)
            gt_cxcywh[:, 0] = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0 / img_w
            gt_cxcywh[:, 1] = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0 / img_h
            gt_cxcywh[:, 2] = (gt_boxes[:, 2] - gt_boxes[:, 0]) / img_w
            gt_cxcywh[:, 3] = (gt_boxes[:, 3] - gt_boxes[:, 1]) / img_h

            b_pred_logits = pred_logits[b] # (N_q, num_classes)
            b_pred_boxes = pred_boxes[b]   # (N_q, 4)

            cost_class = -b_pred_logits.softmax(dim=-1)[:, (gt_labels - 1).long()]
            cost_bbox = torch.cdist(b_pred_boxes, gt_cxcywh, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(b_pred_boxes), gt_boxes / torch.tensor([img_w, img_h, img_w, img_h], device=gt_boxes.device))

            cost_matrix = cost_class + 2.0 * cost_bbox + 2.0 * cost_giou

            # Greedy matching selection
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
        if isinstance(images, list):
            batched_images = torch.stack(images, dim=0)
        else:
            batched_images = images

        B, _, H, W = batched_images.shape

        memory = self._extract_spatiotemporal_features(batched_images)
        if self.use_probabilistic_query:
            queries, kl_loss = self.latent_query_gen(batch_size=B, is_training=self.training)
        else:
            queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
            kl_loss = torch.tensor(0.0, device=batched_images.device)

        ref_points = torch.sigmoid(self.ref_point_head(queries)) # (B, N_q, 3)

        # Transformer Decoder execution
        output = queries
        for layer in self.decoder_layers:
            # 1. Self-Attention
            sa_out, _ = layer["self_attn"](output, output, output)
            output = layer["norm1"](output + sa_out)

            # 2. Spatiotemporal Deformable Cross-Attention
            ca_out = layer["cross_attn"](output, memory, ref_points)
            output = layer["norm2"](output + ca_out)

            # 3. Feed Forward Network
            ffn_out = layer["ffn"](output)
            output = layer["norm3"](output + ffn_out)

        # Prediction Heads
        pred_logits = self.class_head(output)                  # (B, N_q, num_classes)
        pred_boxes = torch.sigmoid(self.box_head(output))      # (B, N_q, 4) normalized cxcywh

        if self.training:
            if targets is None:
                raise ValueError("Targets must be passed during training mode")

            indices = self._hungarian_matching(pred_logits, pred_boxes, targets)

            loss_ce = torch.tensor(0.0, device=batched_images.device)
            loss_bbox = torch.tensor(0.0, device=batched_images.device)
            loss_giou = torch.tensor(0.0, device=batched_images.device)
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
                scores, labels = torch.max(prob[b], dim=-1)
                b_boxes = pred_boxes[b]

                cx, cy, w, h = b_boxes.unbind(-1)
                x1 = (cx - w / 2.0) * W
                y1 = (cy - h / 2.0) * H
                x2 = (cx + w / 2.0) * W
                y2 = (cy + h / 2.0) * H

                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
                labels = labels + 1  # 1-indexed

                results.append({
                    "boxes": boxes_xyxy,
                    "scores": scores,
                    "labels": labels
                })

            return results
