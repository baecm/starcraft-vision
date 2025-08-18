# kbrs.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional


class KBRSConvScorer(nn.Module):
    """
    Memory-friendly scorer using conv2d (no unfold):
      - density: conv over sum(C) with ones kernel
      - centeredness: conv over sum(C) with gaussian kernel
      - mixture: depthwise conv per-channel with ones kernel, sigmoid, then sum across channels
      - projections: mean over group channels, then conv ones
      - mask_channel: multiplicative gain by window-wise mean of mask channel
    """
    def __init__(self,
                 region_size: Tuple[int, int] = (20, 12),
                 weights: Dict[str, float] | None = None,
                 projections: Optional[Dict[str, List[int]]] = None,
                 mixture_tau: float = 2.0,
                 mask_channel: Optional[int] = None,
                 mask_gain: float = 1.0,
                 score_stride: int = 1,
                 downsample_before: int | None = None):
        super().__init__()
        self.kh, self.kw = region_size
        self.weights = weights or {"density": 1.0, "mixture": 1.0, "centeredness": 1.0}
        self.projections = projections or {}
        self.mixture_tau = float(mixture_tau)
        self.mask_channel = mask_channel
        self.mask_gain = float(mask_gain)
        self.score_stride = int(score_stride)
        self.downsample_before = downsample_before

        # ones kernel for density / proj
        ones = torch.ones((1, 1, self.kh, self.kw))
        self.register_buffer("k_ones", ones)

        # gaussian kernel for centeredness
        yy, xx = torch.meshgrid(
            torch.arange(self.kh, dtype=torch.float32),
            torch.arange(self.kw, dtype=torch.float32),
            indexing="ij"
        )
        cy, cx = (self.kh - 1) / 2.0, (self.kw - 1) / 2.0
        sigma = max(self.kh, self.kw) / 4.0
        g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        g = g / (g.sum() + 1e-6)
        self.register_buffer("k_gauss", g.view(1, 1, self.kh, self.kw))

    def _maybe_downsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample_before and self.downsample_before > 1:
            h, w = x.shape[-2:]
            return F.interpolate(x, size=(h // self.downsample_before, w // self.downsample_before),
                                 mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor):
        """
        x: (N, C, H, W)
        returns:
          total: (N, oh, ow)
          comps: dict[name] -> (N, oh, ow)
        """
        x = self._maybe_downsample(x)
        
        k1 = self.k_ones
        kg = self.k_gauss
        if k1.device != x.device or k1.dtype != x.dtype:
            k1 = self.k_ones.to(device=x.device, dtype=x.dtype)
            kg = self.k_gauss.to(device=x.device, dtype=x.dtype)

        N, C, H, W = x.shape
        comps: Dict[str, torch.Tensor] = {}

        # sum over channels once
        x_sum = x.sum(dim=1, keepdim=True)  # (N,1,H,W)

        # density
        den = F.conv2d(x_sum, k1, stride=self.score_stride)
        den = den / (self.kh * self.kw)
        comps["density"] = den.squeeze(1)

        # centeredness
        cen = F.conv2d(x_sum, kg, stride=self.score_stride)
        comps["centeredness"] = cen.squeeze(1)

        # mixture (depthwise)
        w = k1.expand(C, -1, -1, -1).contiguous()  # (C,1,kh,kw) on correct device/dtype
        mix = F.conv2d(x, w, stride=self.score_stride, groups=C)
        mix = torch.sigmoid(mix / self.mixture_tau).sum(dim=1)
        comps["mixture"] = mix

        # projections
        for name, chs in self.projections.items():
            if len(chs) == 0:
                continue
            m = x[:, chs, :, :].mean(dim=1, keepdim=True)               # (N,1,H,W)
            p = F.conv2d(m, self.k_ones, stride=self.score_stride) / (self.kh * self.kw)
            comps[name] = p.squeeze(1)

        # mask channel gain
        if self.mask_channel is not None and 0 <= self.mask_channel < C:
            m = F.conv2d(x[:, self.mask_channel:self.mask_channel+1], self.k_ones, stride=self.score_stride)
            m = m / (self.kh * self.kw)  # window mean 0..?
            # multiplicative gain
            gain = (1.0 + self.mask_gain * m).squeeze(1)  # (N,oh,ow)
        else:
            gain = None

        # weighted sum
        total = 0.0
        for k, v in comps.items():
            total = total + self.weights.get(k, 0.0) * v
        if gain is not None:
            total = total * gain

        return total, comps


class KBRSUnfoldScorer(nn.Module):
    """
    Unfold-based region scorer.
    - density: mean over region
    - centeredness: Gaussian-like weight emphasizing center
    - mixture: combine projections or channels
    """

    def __init__(self,
                 region_size: Tuple[int, int] = (20, 12),
                 score_weights: Dict[str, float] = None,
                 projections: Dict[str, List[int]] = None,
                 mixture_between: Tuple[str, str] = None,
                 mask_channel: int = None):
        super().__init__()
        self.kh, self.kw = region_size
        self.score_weights = score_weights or {"density": 1.0,
                                               "mixture": 1.0,
                                               "centeredness": 1.0}

        # unfold for sliding window
        self.unfold = nn.Unfold(kernel_size=(self.kh, self.kw), stride=1)

        # centeredness mask
        cy, cx = torch.meshgrid(torch.arange(self.kh), torch.arange(self.kw), indexing='ij')
        cy, cx = cy.float(), cx.float()
        center_y, center_x = (self.kh - 1) / 2.0, (self.kw - 1) / 2.0
        dist2 = (cy - center_y) ** 2 + (cx - center_x) ** 2
        sigma2 = (self.kh * self.kw) / 4.0
        center_mask = torch.exp(-dist2 / (2 * sigma2))
        self.register_buffer("center_mask", center_mask)

        # channel group definitions
        self.projections = projections or {}
        self.mixture_between = mixture_between
        self.mask_channel = mask_channel

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: (N, C, H, W)
        Returns:
            total_score_map: (N, oh, ow)
            comp_maps: dict[type] -> (N, oh, ow)
        """
        N, C, H, W = x.shape
        patches = self.unfold(x)  # (N, C*kh*kw, L)
        L = patches.size(-1)
        oh = H - self.kh + 1
        ow = W - self.kw + 1

        patches = patches.view(N, C, self.kh * self.kw, L)

        comp_maps = {}

        # density
        density_map = patches.mean(dim=2)  # (N, C, L)
        density_map = density_map.mean(dim=1).view(N, oh, ow)
        comp_maps["density"] = density_map

        # centeredness
        cmask = self.center_mask.view(1, 1, self.kh * self.kw, 1)
        centeredness_map = (patches * cmask).mean(dim=2)
        centeredness_map = centeredness_map.mean(dim=1).view(N, oh, ow)
        comp_maps["centeredness"] = centeredness_map

        # projections
        for pname, channels in self.projections.items():
            proj_map = patches[:, channels].mean(dim=1).mean(dim=1).view(N, oh, ow)
            comp_maps[pname] = proj_map

        # mixture between projections
        if self.mixture_between is not None:
            a, b = self.mixture_between
            if a in comp_maps and b in comp_maps:
                comp_maps["mixture"] = (comp_maps[a] + comp_maps[b]) / 2.0

        # mask channel
        if self.mask_channel is not None and 0 <= self.mask_channel < C:
            mask_val = patches[:, self.mask_channel].mean(dim=1).view(N, oh, ow)
            comp_maps["mask"] = mask_val

        # weighted sum
        total_score_map = 0.0
        for k, v in comp_maps.items():
            w = self.score_weights.get(k, 0.0)
            total_score_map = total_score_map + w * v

        return total_score_map, comp_maps
