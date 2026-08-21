# src/models/backbones/spatiotemporal_encoder.py
from __future__ import annotations

import torch
import torch.nn as nn


class SpatioTemporalEncoder(nn.Module):
    """
    3D CNN / Spatio-Temporal Encoder for Multi-Channel State Tensors X_t.
    Input: (B, T, C_in, H, W)
    Output: Feature maps of shape (B, T, C_feat, H_out, W_out)
    """

    def __init__(
        self,
        in_channels: int = 4,
        feat_dim: int = 128,
        downsample_factor: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_dim = feat_dim

        # 3D Convolution for joint spatio-temporal feature extraction
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                feat_dim // 2,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                stride=(1, 2, 2),
            ),
            nn.BatchNorm3d(feat_dim // 2),
            nn.ReLU(inplace=True),
        )

        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(
                feat_dim // 2,
                feat_dim,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                stride=(1, 2, 2) if downsample_factor >= 4 else (1, 1, 1),
            ),
            nn.BatchNorm3d(feat_dim),
            nn.ReLU(inplace=True),
        )

        # Spatial refinement residual block
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        returns: (B, T, feat_dim, H_out, W_out)
        """
        B, T, C, H, W = x.shape
        # Permute for Conv3d: (B, C, T, H, W)
        x_3d = x.permute(0, 2, 1, 3, 4)

        f_3d = self.conv3d_1(x_3d)
        f_3d = self.conv3d_2(f_3d)

        # Permute back to (B, T, feat_dim, H_out, W_out)
        _, C_f, T_f, H_f, W_f = f_3d.shape
        f_2d = f_3d.permute(0, 2, 1, 3, 4).contiguous()

        # Reshape to 2D for spatial refinement
        f_flat = f_2d.view(B * T, C_f, H_f, W_f)
        f_refined = f_flat + self.spatial_refine(f_flat)
        f_out = f_refined.view(B, T, C_f, H_f, W_f)

        return f_out
