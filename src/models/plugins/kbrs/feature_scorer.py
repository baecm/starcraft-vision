# src/models/plugins/kbrs/feature_scorer.py
"""
The KBRS scorer exactly as it trained the models of the saliency-prior paper
(ToG-2026-0161): src/model/kbrs.py and src/model/kbrs_kernel.py at e949efa.

It differs from scorer.py, which was rewritten in the August refactor and has
never trained a reported model: that one applies a sigmoid first and scores
mixture as 4 * d_A * d_B, where this one sums raw activations and scores it as
4p(1-p) with p = A / (A + B), the form the paper states.

The input is the detector's feature map, not the game-state tensor. The
projections name channel groups by their game-state indices (player 1 = 0-3,
player 2 = 4-7, expanded over the stacked window) and are applied to the
feature channels with the same indices, which is what the original code did.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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


class KBRSFeatureScorer(nn.Module):
    """Density / centeredness / mixture on a (N, C, H, W) feature map -> (N, oh, ow)."""

    def __init__(
        self,
        region_size: Tuple[int, int] = (20, 12),
        weights: Optional[Dict[str, float]] = None,
        projections: Optional[Dict[str, List[int]]] = None,
        mixture_mode: str = "confusion",
        mixture_power: float = 1.0,
        score_stride: int = 1,
        downsample_before: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.kh, self.kw = region_size
        self.weights: Dict[str, float] = dict(
            weights or {"density": 1.0, "mixture": 1.0, "centeredness": 1.0}
        )
        self.projections: Dict[str, List[int]] = projections or {}
        self.mixture_mode = str(mixture_mode)
        self.mixture_power = float(mixture_power)
        self.score_stride = int(score_stride)
        self.downsample_before = downsample_before
        self.register_buffer("k_ones_f32", make_ones_kernel(self.kh, self.kw), persistent=False)
        self.register_buffer("k_center_f32", make_center_kernel(self.kh, self.kw), persistent=False)

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

    def _density(self, x: torch.Tensor) -> torch.Tensor:
        x_sum = x.sum(dim=1, keepdim=True)
        k1 = self._cast_buf(self.k_ones_f32, x_sum)
        den = F.conv2d(x_sum, k1, stride=self.score_stride)
        return (den / float(self.kh * self.kw)).squeeze(1)

    def _centeredness(self, x: torch.Tensor) -> torch.Tensor:
        x_sum = x.sum(dim=1, keepdim=True)
        kc = self._cast_buf(self.k_center_f32, x_sum)
        cen = F.conv2d(x_sum, kc, stride=self.score_stride)
        return (cen / (kc.sum() + 1e-6)).squeeze(1)

    def _mixture(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        extra: Dict[str, torch.Tensor] = {}
        idx_a = self.projections.get("A", [])
        idx_b = self.projections.get("B", [])
        if idx_a and idx_b:
            # Each group is summed into one channel, windowed, and mixed as 4p(1-p).
            xa = x[:, idx_a, :, :].sum(dim=1, keepdim=True)
            xb = x[:, idx_b, :, :].sum(dim=1, keepdim=True)
            k1 = self._cast_buf(self.k_ones_f32, xa)
            a = torch.nan_to_num(F.conv2d(xa, k1, stride=self.score_stride), nan=0.0, posinf=0.0, neginf=0.0)
            b = torch.nan_to_num(F.conv2d(xb, k1, stride=self.score_stride), nan=0.0, posinf=0.0, neginf=0.0)
            eps = 1e-6
            p = (a / (a + b).clamp_min(eps)).clamp(0.0, 1.0)
            if self.mixture_mode == "entropy":
                conf = -(p * p.clamp_min(eps).log() + (1 - p) * (1 - p).clamp_min(eps).log())
                conf = conf / torch.log(torch.tensor(2.0, device=x.device, dtype=x.dtype))
            else:
                conf = 4.0 * p * (1.0 - p)
            if self.mixture_power != 1.0:
                conf = conf.clamp_(0, 1).pow(self.mixture_power)
            extra["proj_A"] = a.squeeze(1)
            extra["proj_B"] = b.squeeze(1)
            return conf.squeeze(1), extra
        # Without both groups the original fell back to the channel-mean window sum.
        k1 = self._cast_buf(self.k_ones_f32, x)
        act = F.conv2d(x, k1, stride=self.score_stride).mean(dim=1)
        return act, extra

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.dim() != 4:
            raise ValueError(f"KBRSFeatureScorer expects (N, C, H, W), got {tuple(x.shape)}")
        x = self._maybe_downsample(x)
        comps: Dict[str, torch.Tensor] = {}
        if self.weights.get("density", 0.0) != 0.0:
            comps["density"] = self._density(x)
        if self.weights.get("centeredness", 0.0) != 0.0:
            comps["centeredness"] = self._centeredness(x)
        if self.weights.get("mixture", 0.0) != 0.0:
            mix, extra = self._mixture(x)
            comps["mixture"] = mix
            comps.update(extra)
        if not any(k in comps for k in ("density", "centeredness", "mixture")):
            raise ValueError("KBRSFeatureScorer: every component weight is zero")

        score = torch.zeros_like(next(iter(comps.values())))
        for k in ("density", "centeredness", "mixture"):
            if k in comps:
                score = score + self.weights[k] * comps[k]
        for k, v in list(comps.items()):
            comps[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e3, 1e3)
        return score, comps
