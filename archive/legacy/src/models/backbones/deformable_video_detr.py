# src/models/backbones/deformable_video_detr.py
from __future__ import annotations

from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatiotemporal_encoder import SpatioTemporalEncoder
from ..plugins.cvae_query import CVAELatentQueryInjector


class DeformableVideoDETRDecoder(nn.Module):
    """
    Deformable Video DETR Decoder with Video Temporal Query Propagation.
    Applies temporal self-attention across frame queries and deformable spatial cross-attention over feature maps.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_queries: int = 3,
        num_heads: int = 8,
        num_decoder_layers: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_heads = num_heads

        # Base learnable query embeddings
        self.base_queries = nn.Parameter(torch.randn(num_queries, embed_dim))

        # Temporal Query Self-Attention over sequence T
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

        # Deformable Spatial Cross-Attention layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 2,
                batch_first=True,
            )
            for _ in range(num_decoder_layers)
        ])

        # Prediction Heads
        self.box_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 4),
            nn.Sigmoid(),  # Normalized boxes [cx, cy, w, h] in range [0, 1]
        )

        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        feat_seq: torch.Tensor,
        latent_queries: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        feat_seq: (B, T, C_feat, H_out, W_out)
        latent_queries: Optional (B, T, K, embed_dim) or (B, K, embed_dim)

        Returns:
          - 'pred_boxes': (B, T, K, 4) normalized [cx, cy, w, h]
          - 'pred_scores': (B, T, K)
          - 'query_embeddings': (B, T, K, embed_dim)
        """
        B, T, C, H, W = feat_seq.shape
        K = self.num_queries

        # Initialize queries across (B, T, K, embed_dim)
        queries = self.base_queries.view(1, 1, K, self.embed_dim).expand(B, T, K, self.embed_dim).clone()

        if latent_queries is not None:
            if latent_queries.ndim == 3:  # (B, K, embed_dim)
                queries = queries + latent_queries.unsqueeze(1)
            elif latent_queries.ndim == 4:  # (B, T, K, embed_dim)
                queries = queries + latent_queries

        # Temporal Query Self-Attention (sequence continuity)
        # Reshape to (B * K, T, embed_dim) for temporal attention per query index
        queries_perm = queries.permute(0, 2, 1, 3).reshape(B * K, T, self.embed_dim)
        temp_queries, _ = self.temporal_attn(queries_perm, queries_perm, queries_perm)
        queries = temp_queries.view(B, K, T, self.embed_dim).permute(0, 2, 1, 3).contiguous()

        # Spatial Cross-Attention per frame t
        pred_boxes_list = []
        pred_scores_list = []

        for t in range(T):
            f_t = feat_seq[:, t]  # (B, C, H, W)
            q_t = queries[:, t]   # (B, K, embed_dim)

            # Flatten spatial feature map as memory key/value: (B, H*W, C)
            memory_t = f_t.flatten(2).permute(0, 2, 1)

            out_q = q_t
            for layer in self.decoder_layers:
                out_q = layer(out_q, memory_t)

            boxes_t = self.box_head(out_q)        # (B, K, 4)
            scores_t = self.score_head(out_q).squeeze(-1)  # (B, K)

            pred_boxes_list.append(boxes_t)
            pred_scores_list.append(scores_t)

        pred_boxes = torch.stack(pred_boxes_list, dim=1)    # (B, T, K, 4)
        pred_scores = torch.stack(pred_scores_list, dim=1)  # (B, T, K)

        return {
            "pred_boxes": pred_boxes,
            "pred_scores": pred_scores,
            "query_embeddings": queries,
        }


class ProbabilisticVideoDETR(nn.Module):
    """
    Modular Video DETR Architecture combining:
      1. SpatioTemporalEncoder (3D CNN)
      2. CVAELatentQueryInjector (Probabilistic Latent Query Injection)
      3. DeformableVideoDETRDecoder (Temporal + Spatial Cross-Attention)
      4. Heatmap Head (Continuous Importance Map S(x,y,t))
    """

    def __init__(
        self,
        in_channels: int = 4,
        feat_dim: int = 128,
        num_raters: int = 5,
        latent_dim: int = 64,
        num_queries: int = 3,
        num_heads: int = 8,
        num_decoder_layers: int = 3,
        grid_size: Tuple[int, int] = (128, 128),
        use_cvae: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_dim = feat_dim
        self.num_queries = num_queries
        self.grid_size = grid_size
        self.use_cvae = use_cvae

        self.encoder = SpatioTemporalEncoder(
            in_channels=in_channels, feat_dim=feat_dim
        )

        if self.use_cvae:
            self.cvae_injector = CVAELatentQueryInjector(
                feat_dim=feat_dim,
                num_raters=num_raters,
                latent_dim=latent_dim,
                embed_dim=feat_dim,
                num_queries=num_queries,
            )
        else:
            self.cvae_injector = None

        self.decoder = DeformableVideoDETRDecoder(
            embed_dim=feat_dim,
            num_queries=num_queries,
            num_heads=num_heads,
            num_decoder_layers=num_decoder_layers,
        )

        # Heatmap prediction head predicting continuous importance map S(x, y, t)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(feat_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        human_heatmaps: Optional[torch.Tensor] = None,
        use_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        x: Spatio-temporal state tensor (B, T, C_in, H, W)
        human_heatmaps: Optional 5-rater gaze heatmaps (B, T, 5, H, W)

        Returns dict:
          - 'pred_boxes': (B, T, K, 4) in [cx, cy, w, h] normalized format
          - 'pred_scores': (B, T, K)
          - 'pred_heatmap': (B, T, 1, H_grid, W_grid) continuous importance heatmap S(x,y,t)
          - 'latent_out': dict with z, mu, logvar, prior_mu, prior_logvar (if use_cvae)
        """
        B, T, C, H, W = x.shape

        # 1. Spatio-Temporal Feature Encoding
        feat_seq = self.encoder(x)  # (B, T, feat_dim, H_out, W_out)

        # 2. CVAE Latent Query Injection across sequence T
        latent_queries = None
        aggregated_latent = None

        if self.use_cvae and self.cvae_injector is not None:
            latent_queries_list = []
            cvae_outputs_list = []

            for t in range(T):
                f_t = feat_seq[:, t]
                h_t = human_heatmaps[:, t] if human_heatmaps is not None else None

                cvae_out = self.cvae_injector(
                    feat_map=f_t, human_heatmaps=h_t, use_posterior=use_posterior
                )
                latent_queries_list.append(cvae_out["latent_query"])
                cvae_outputs_list.append(cvae_out)

            latent_queries = torch.stack(latent_queries_list, dim=1)  # (B, T, K, feat_dim)

            # Aggregate latent stats for loss calculation
            aggregated_latent = {
                "mu": torch.stack([c["mu"] for c in cvae_outputs_list], dim=1),
                "logvar": torch.stack([c["logvar"] for c in cvae_outputs_list], dim=1),
                "prior_mu": torch.stack([c["prior_mu"] for c in cvae_outputs_list], dim=1),
                "prior_logvar": torch.stack([c["prior_logvar"] for c in cvae_outputs_list], dim=1),
            }

        # 3. Deformable Video DETR Decoder
        dec_out = self.decoder(feat_seq=feat_seq, latent_queries=latent_queries)

        # 4. Continuous Importance Heatmap Estimation Head S(x, y, t)
        feat_flat = feat_seq.view(B * T, self.feat_dim, feat_seq.shape[-2], feat_seq.shape[-1])
        heatmap_flat = self.heatmap_head(feat_flat)  # (B*T, 1, H_out, W_out)
        heatmap_flat = F.interpolate(
            heatmap_flat, size=self.grid_size, mode="bilinear", align_corners=False
        )
        pred_heatmap = heatmap_flat.view(B, T, 1, self.grid_size[0], self.grid_size[1])

        out = {
            "pred_boxes": dec_out["pred_boxes"],
            "pred_scores": dec_out["pred_scores"],
            "pred_heatmap": pred_heatmap,
        }
        if aggregated_latent is not None:
            out["latent_out"] = aggregated_latent

        return out
