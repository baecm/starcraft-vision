from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class DensityPeakHead(nn.Module):
    """
    Density Peak Head & Delta Head plugin for heatmap-based architectures (CenterNet).
    Predicts density factor and spatial offsets (delta) for peak-finding decoding.
    """
    def __init__(self, in_channels: int, head_conv: int = 64):
        super().__init__()
        self.density_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1)
        )
        self.delta_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1)
        )

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred_density = self.density_head(feat)
        pred_delta = self.delta_head(feat)
        return pred_density, pred_delta
