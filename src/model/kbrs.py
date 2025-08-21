# src/model/kbrs.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from utils.logger import Logger

# =========================
# Conv2d 기반 KBRS Scorer
# =========================
class KBRSConvScorer(nn.Module):
    """
    Conv2d로 density/mixture/centeredness를 계산해 (N, oh, ow) score_map을 반환.
    - region_size: (kh, kw) 윈도우
    - weights: 각 컴포넌트 합성 가중치 dict
    - projections: dict{name -> [channel indices]}  (예: {"A":[0,1,2,3], "B":[4,5,6,7]})
    - mixture_mode: "confusion"(4p(1-p)) | "entropy" | "agreement"(fallback로 confusion 사용)
    - mixture_power: confusion 값을 거듭제곱(>1이면 중앙부(p≈0.5) 더 강조)
    - score_stride: conv stride
    - downsample_before: None 또는 {"type":"avg","stride":2} 식의 프리다운샘플
    """
    def __init__(
        self,
        region_size: Tuple[int, int] = (20, 12),
        weights: Optional[Dict[str, float]] = None,
        projections: Optional[Dict[str, List[int]]] = None,
        mixture_tau: float = 2.0,  # (fallback용)
        mixture_mode: str = "confusion",
        mixture_power: float = 1.0,
        mask_channel: Optional[int] = None,  # (미사용: gate는 모델에서 처리)
        mask_gain: float = 1.0,
        score_stride: int = 1,
        downsample_before: Optional[Dict] = None,
    ):
        super().__init__()
        self.kh, self.kw = region_size
        self.weights = dict(weights or {"density": 1.0, "mixture": 1.0, "centeredness": 1.0})
        self.projections = projections or {}
        self.mixture_tau = float(mixture_tau)
        self.mixture_mode = str(mixture_mode)
        self.mixture_power = float(mixture_power)
        self.score_stride = int(score_stride)
        self.downsample_before = downsample_before

        # ones kernel / gaussian kernel 은 buffer로 잡되, 사용시 입력 dtype/device로 캐스팅
        ones = torch.ones((1, 1, self.kh, self.kw), dtype=torch.float32)
        self.register_buffer("k_ones_f32", ones, persistent=False)

        # centeredness weight (2D gaussian)
        yy, xx = torch.meshgrid(
            torch.arange(self.kh, dtype=torch.float32),
            torch.arange(self.kw, dtype=torch.float32),
            indexing="ij"
        )
        cy, cx = (self.kh - 1) / 2.0, (self.kw - 1) / 2.0
        sigma = self.kh / 4.0
        w = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (sigma ** 2)))
        self.register_buffer("k_center_f32", w.view(1, 1, self.kh, self.kw), persistent=False)

    def _cast_buf(self, buf: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return buf.to(device=ref.device, dtype=ref.dtype)

    def _maybe_downsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample_before is None:
            return x
        t = self.downsample_before.get("type", "avg").lower()
        s = int(self.downsample_before.get("stride", 2))
        if t == "avg":
            return F.avg_pool2d(x, kernel_size=s, stride=s)
        elif t == "max":
            return F.max_pool2d(x, kernel_size=s, stride=s)
        else:
            return x

    def _density(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,C,H,W) -> 채널합에 ones-kernel conv => 윈도우 합/평균
        x_sum = x.sum(dim=1, keepdim=True)                    # (N,1,H,W)
        k1 = self._cast_buf(self.k_ones_f32, x)
        den = F.conv2d(x_sum, k1, stride=self.score_stride)   # (N,1,oh,ow)
        den = den / float(self.kh * self.kw)
        return den.squeeze(1)                                 # (N,oh,ow)

    def _centeredness(self, x: torch.Tensor) -> torch.Tensor:
        # 채널합 후 가우시안 커널로 conv
        x_sum = x.sum(dim=1, keepdim=True)                    # (N,1,H,W)
        kc = self._cast_buf(self.k_center_f32, x)
        cen = F.conv2d(x_sum, kc, stride=self.score_stride)   # (N,1,oh,ow)
        # 평균 정규화(선택): 스케일 안정화
        cen = cen / (kc.sum() + 1e-6)
        return cen.squeeze(1)                                 # (N,oh,ow)

    def _mixture_from_projections(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # projection A/B가 있으면 A/B 채널합 -> ones conv -> p=A/(A+B), confusion/entropy 등 계산
        comp_extra = {}
        if "A" in self.projections and "B" in self.projections:
            idxA = self.projections["A"]
            idxB = self.projections["B"]
            xA = x[:, idxA, :, :].sum(dim=1, keepdim=True)         # (N,1,H,W)
            xB = x[:, idxB, :, :].sum(dim=1, keepdim=True)         # (N,1,H,W)

            k1 = self._cast_buf(self.k_ones_f32, x)
            A = F.conv2d(xA, k1, stride=self.score_stride)         # (N,1,oh,ow)
            B = F.conv2d(xB, k1, stride=self.score_stride)         # (N,1,oh,ow)

            eps = 1e-6
            den = (A + B).clamp_min(eps)
            p = A / den                                            # (N,1,oh,ow)

            mode = self.mixture_mode
            if mode == "entropy":
                conf = -(p * (p.clamp_min(eps).log()) +
                         (1 - p) * ((1 - p).clamp_min(eps).log())) / torch.log(torch.tensor(2.0, device=x.device, dtype=x.dtype))
            else:
                # "confusion" (기본) 또는 기타 → confusion로 fallback
                conf = 4.0 * p * (1.0 - p)

            if self.mixture_power != 1.0:
                conf = conf.clamp_(0, 1).pow(self.mixture_power)

            comp_extra["proj_A"] = A.squeeze(1)
            comp_extra["proj_B"] = B.squeeze(1)
            comp_extra["proj_mixture"] = conf.squeeze(1)

            return conf.squeeze(1), comp_extra

        # projections가 없을 때의 fallback mixture (활성 채널 수 근사)
        # 각 채널에 ones conv 후 >0 근사(sigmoid/tau), 채널합
        N, C, H, W = x.shape
        kC = self._cast_buf(self.k_ones_f32, x).expand(C, 1, self.kh, self.kw)   # (C,1,kh,kw)
        # 그룹 conv로 각 채널 별 윈도우 합
        ch_sum = F.conv2d(x, kC, stride=self.score_stride, groups=C)             # (N,C,oh,ow)
        act = torch.sigmoid(ch_sum / self.mixture_tau)                           # (N,C,oh,ow)
        mix = act.mean(dim=1)                                                    # (N,oh,ow)  (mean or sum)
        return mix, comp_extra

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: (N,C,H,W)
        return:
          score_map: (N, oh, ow)
          comp_maps: dict[type] -> (N, oh, ow)  (density/mixture/centeredness/proj_A/proj_B/proj_mixture...)
        """
        assert x.dim() == 4
        x = self._maybe_downsample(x)

        comp_maps: Dict[str, torch.Tensor] = {}

        # density
        if self.weights.get("density", 0.0) != 0.0:
            comp_maps["density"] = self._density(x)

        # centeredness
        if self.weights.get("centeredness", 0.0) != 0.0:
            comp_maps["centeredness"] = self._centeredness(x)

        # mixture
        if self.weights.get("mixture", 0.0) != 0.0:
            mix_map, extra = self._mixture_from_projections(x)
            comp_maps["mixture"] = mix_map
            comp_maps.update(extra)  # proj_A/proj_B/proj_mixture(=confusion)

        # 합성
        total = 0.0
        for k, v in comp_maps.items():
            w = float(self.weights.get(k, 0.0))
            if w != 0.0:
                total = total + w * v
        score_map = total if isinstance(total, torch.Tensor) else torch.zeros_like(next(iter(comp_maps.values())))

        return score_map, comp_maps


# =========================
# (옵션) Unfold 기반 Scorer
# =========================
class KBRSUnfoldScorer(nn.Module):
    """
    메모리 사용이 커서 특별한 이유가 없으면 KBRSConvScorer를 이용하세요.
    여기서는 간단한 density/mixture/centeredness만 유지합니다.
    """
    def __init__(
        self,
        region_size: Tuple[int, int] = (20, 12),
        score_weights: Optional[Dict[str, float]] = None,
        projections: Optional[Dict[str, List[int]]] = None,
        mixture_between: Optional[Tuple[str, str]] = None,
        mask_channel: Optional[int] = None,
    ):
        super().__init__()
        self.kh, self.kw = region_size
        self.w = dict(score_weights or {"density": 1.0, "mixture": 1.0, "centeredness": 1.0})
        self.projections = projections or {}

    def forward(self, x: torch.Tensor):
        # 간단 버전: conv 스코어러 권장
        conv = KBRSConvScorer(region_size=(self.kh, self.kw), weights=self.w, projections=self.projections)
        conv = conv.to(x.device, dtype=x.dtype)
        return conv(x)
