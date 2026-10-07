from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_ones_kernel(kh: int, kw: int) -> torch.Tensor:
    return torch.ones((1, 1, kh, kw), dtype=torch.float32)

def make_center_kernel(kh: int, kw: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(kh, dtype=torch.float32),
        torch.arange(kw, dtype=torch.float32),
        indexing="ij",
    )
    cy, cx = (kh - 1) / 2.0, (kw - 1) / 2.0
    sigma = kh / 4.0
    w = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (sigma ** 2)))
    return w.view(1, 1, kh, kw)

class KBRSConvScorer(nn.Module):
    """
    Conv2d-based KBRS Scorer that computes density / mixture / centeredness
    to output an (N, oh, ow) score_map tensor.
    """
    def __init__(
        self,
        region_size: Tuple[int, int] = (20, 12),
        weights: Optional[Dict[str, float]] = None,
        projections: Optional[Dict[str, List[int]]] = None,
        mixture_tau: float = 2.0,
        mixture_mode: str = "confusion",
        mixture_power: float = 1.0,
        mask_channel: Optional[int] = None,
        mask_gain: float = 1.0,
        score_stride: int = 1,
        downsample_before: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.kh, self.kw = region_size
        self.weights: Dict[str, float] = dict(
            weights or {"density": 1.0, "mixture": 1.0, "centeredness": 1.0}
        )
        self.projections: Dict[str, List[int]] = projections or {}
        self.mixture_tau = float(mixture_tau)
        self.mixture_mode = str(mixture_mode)
        self.mixture_power = float(mixture_power)
        self.mask_channel = mask_channel
        self.mask_gain = float(mask_gain)
        self.score_stride = int(score_stride)
        self.downsample_before = downsample_before

        ones = make_ones_kernel(self.kh, self.kw)
        self.register_buffer("k_ones_f32", ones, persistent=False)

        center = make_center_kernel(self.kh, self.kw)
        self.register_buffer("k_center_f32", center, persistent=False)

    def _cast_buf(self, buf: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return buf.to(device=ref.device, dtype=ref.dtype)

    def _maybe_downsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample_before is None:
            return x

        t = str(self.downsample_before.get("type", "avg")).lower()
        s = int(self.downsample_before.get("stride", 2))

        if s <= 1:
            return x

        if t == "avg":
            return F.avg_pool2d(x, kernel_size=s, stride=s)
        if t == "max":
            return F.max_pool2d(x, kernel_size=s, stride=s)
        return x

    def forward(
        self,
        feat_map: torch.Tensor,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if feat_map.dim() != 4:
            raise ValueError(f"feat_map must be 4D (N, C, H, W), got {feat_map.shape}")

        x = self._maybe_downsample(feat_map)
        N, C, H, W = x.shape

        if H < self.kh or W < self.kw:
            raise ValueError(f"Feature map size ({H},{W}) smaller than region_size ({self.kh},{self.kw})")

        w_dict = self.weights if weights is None else dict(weights)
        w_d = float(w_dict.get("density", 1.0))
        w_m = float(w_dict.get("mixture", 1.0))
        w_c = float(w_dict.get("centeredness", 1.0))

        p_map = torch.sigmoid(x)
        ones_k = self._cast_buf(self.k_ones_f32, x)
        cent_k = self._cast_buf(self.k_center_f32, x)
        win_area = float(self.kh * self.kw)

        proj_A = self.projections.get("A", [])
        proj_B = self.projections.get("B", [])

        if len(proj_A) > 0:
            sum_A = p_map[:, proj_A, :, :].sum(dim=1, keepdim=True)
            cnt_A = float(len(proj_A))
        else:
            sum_A = p_map.sum(dim=1, keepdim=True)
            cnt_A = float(C)

        if len(proj_B) > 0:
            sum_B = p_map[:, proj_B, :, :].sum(dim=1, keepdim=True)
            cnt_B = float(len(proj_B))
        else:
            sum_B = torch.zeros_like(sum_A)
            cnt_B = 0.0

        p_total = (sum_A + sum_B) / max(1.0, (cnt_A + cnt_B))

        area_sum = F.conv2d(p_total, ones_k, stride=self.score_stride)
        density_map = area_sum.squeeze(1) / win_area

        cent_sum = F.conv2d(p_total, cent_k, stride=self.score_stride)
        k_center_sum = float(cent_k.sum().item())
        centeredness_map = cent_sum.squeeze(1) / max(1e-6, k_center_sum)

        if cnt_B > 0:
            den_A = F.conv2d(sum_A / cnt_A, ones_k, stride=self.score_stride).squeeze(1) / win_area
            den_B = F.conv2d(sum_B / cnt_B, ones_k, stride=self.score_stride).squeeze(1) / win_area

            mode = self.mixture_mode.lower()
            if mode == "entropy":
                eps = 1e-6
                total_den = (den_A + den_B).clamp_min(eps)
                pA = (den_A / total_den).clamp(eps, 1.0 - eps)
                pB = (den_B / total_den).clamp(eps, 1.0 - eps)
                entropy = -(pA * torch.log2(pA) + pB * torch.log2(pB))
                mixture_map = (1.0 - entropy).clamp(0.0, 1.0)
            else:
                mixture_map = 4.0 * den_A * den_B
                mixture_map = mixture_map.clamp(0.0, 1.0)

            if self.mixture_power != 1.0:
                mixture_map = mixture_map.pow(self.mixture_power)
        else:
            mixture_map = torch.zeros_like(density_map)

        score_map = w_d * density_map + w_m * mixture_map + w_c * centeredness_map
        components = {
            "density": density_map,
            "mixture": mixture_map,
            "centeredness": centeredness_map,
        }
        return score_map, components
