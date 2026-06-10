# src/model/KBRSConvScorer.py
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import Logger
from .kbrs_kernel import make_ones_kernel, make_center_kernel


# =========================
# Conv2d 기반 KBRS Scorer
# =========================
class KBRSConvScorer(nn.Module):
    """
    Conv2d로 density / mixture / centeredness를 계산해서 (N, oh, ow) score_map을 반환하는 모듈.

    Args
    ----
    region_size:
        (kh, kw) 윈도우 크기.
    weights:
        {"density": w_d, "mixture": w_m, "centeredness": w_c} 형태의 가중치 dict.
    projections:
        {"A": [채널 인덱스...], "B": [채널 인덱스...]} 식으로 mixture 계산에 사용할 채널 그룹.
    mixture_tau:
        (예전 fallback용) 현재 confusion/entropy 정의에서는 직접 사용하지 않지만 인터페이스 유지용.
    mixture_mode:
        "confusion" | "entropy" 중 선택. 기본은 "confusion".
    mixture_power:
        mixture map에 대한 pow. >1이면 중앙(p≈0.5) 근처를 더 강조.
    mask_channel, mask_gain:
        현재 이 모듈 내부에서는 사용하지 않고, 상위 모델에서 gate 용도로 처리.
    score_stride:
        Conv stride.
    downsample_before:
        {"type": "avg"|"max", "stride": s} 형태. 입력 feature에 사전 downsample을 적용할 때 사용.
    """

    def __init__(
        self,
        region_size: Tuple[int, int] = (20, 12),
        weights: Optional[Dict[str, float]] = None,
        projections: Optional[Dict[str, List[int]]] = None,
        mixture_tau: float = 2.0,  # (fallback용, 인터페이스 유지)
        mixture_mode: str = "confusion",
        mixture_power: float = 1.0,
        mask_channel: Optional[int] = None,  # (미사용: gate는 모델에서 처리)
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
        print(f"[DEBUG KBRS] Received projections: {self.projections}")
        print(f"[DEBUG KBRS] Type: {type(self.projections)}")
        self.mixture_tau = float(mixture_tau)
        self.mixture_mode = str(mixture_mode)
        self.mixture_power = float(mixture_power)
        self.mask_channel = mask_channel
        self.mask_gain = float(mask_gain)
        self.score_stride = int(score_stride)
        self.downsample_before = downsample_before

        # ones / gaussian kernel 은 공통 helper에서 생성하고,
        # 사용 시 _cast_buf 로 dtype/device 를 맞춰서 사용한다.
        ones = make_ones_kernel(self.kh, self.kw)
        self.register_buffer("k_ones_f32", ones, persistent=False)

        center = make_center_kernel(self.kh, self.kw)
        self.register_buffer("k_center_f32", center, persistent=False)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _cast_buf(self, buf: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """buffer를 ref tensor의 device / dtype에 맞춰서 캐스팅."""
        return buf.to(device=ref.device, dtype=ref.dtype)

    def _maybe_downsample(self, x: torch.Tensor) -> torch.Tensor:
        """사전 downsample 옵션이 있을 때 적용."""
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

    # ------------------------------------------------------------------
    # components: density / centeredness / mixture
    # ------------------------------------------------------------------
    def _density(self, x: torch.Tensor) -> torch.Tensor:
        """
        density: 윈도우 안의 채널합 평균.

        x: (N, C, H, W)
        return: (N, oh, ow)
        """
        x_sum = x.sum(dim=1, keepdim=True)  # (N,1,H,W)
        k1 = self._cast_buf(self.k_ones_f32, x_sum)
        den = F.conv2d(x_sum, k1, stride=self.score_stride)  # (N,1,oh,ow)
        den = den / float(self.kh * self.kw)
        return den.squeeze(1)  # (N,oh,ow)

    def _centeredness(self, x: torch.Tensor) -> torch.Tensor:
        """
        centeredness: 2D Gaussian kernel로 weighted average.

        x: (N, C, H, W)
        return: (N, oh, ow)
        """
        x_sum = x.sum(dim=1, keepdim=True)  # (N,1,H,W)
        kc = self._cast_buf(self.k_center_f32, x_sum)
        cen = F.conv2d(x_sum, kc, stride=self.score_stride)  # (N,1,oh,ow)
        cen = cen / (kc.sum() + 1e-6)
        return cen.squeeze(1)

    def _mixture_from_projections(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        projections["A"], projections["B"] 채널 그룹이 있을 경우:
          - A, B 채널 합에 ones-kernel conv 적용
          - p = A/(A+B) 로부터 confusion/entropy 기반 mixture 계산

        projections 가 없으면:
          - 각 채널에 ones-kernel conv 후 채널 평균으로 fallback mixture.
        """
        comp_extra: Dict[str, torch.Tensor] = {}

        # ------------------------
        # 1) projection A/B가 있을 때
        # ------------------------
        if "A" in self.projections and "B" in self.projections:
            idxA = self.projections["A"]
            idxB = self.projections["B"]

            if len(idxA) == 0 or len(idxB) == 0:
                Logger().warning(
                    "[KBRSConvScorer] projections['A'] 또는 ['B']가 비어 있음. "
                    "fallback mixture 로 전환합니다."
                )
            else:
                # (N,1,H,W)
                xA = x[:, idxA, :, :].sum(dim=1, keepdim=True)
                xB = x[:, idxB, :, :].sum(dim=1, keepdim=True)

                k1 = self._cast_buf(self.k_ones_f32, xA)
                A = F.conv2d(xA, k1, stride=self.score_stride)  # (N,1,oh,ow)
                B = F.conv2d(xB, k1, stride=self.score_stride)  # (N,1,oh,ow)

                # A, B 너무 큰 값/NaN 방지용
                A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
                B = torch.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)

                eps = 1e-6
                den = (A + B).clamp_min(eps)
                p = (A / den).clamp(0.0, 1.0)  # (N,1,oh,ow)

                mode = self.mixture_mode
                if mode == "entropy":
                    # binary entropy (log base 2)
                    conf = -(p * (p.clamp_min(eps).log()) +
                             (1 - p) * ((1 - p).clamp_min(eps).log()))
                    conf = conf / torch.log(
                        torch.tensor(2.0, device=x.device, dtype=x.dtype)
                    )
                else:
                    # "confusion" (기본) 또는 기타 → confusion으로 fallback
                    conf = 4.0 * p * (1.0 - p)

                if self.mixture_power != 1.0:
                    conf = conf.clamp_(0, 1).pow(self.mixture_power)

                comp_extra["proj_A"] = A.squeeze(1)
                comp_extra["proj_B"] = B.squeeze(1)
                comp_extra["proj_mixture"] = conf.squeeze(1)

                return conf.squeeze(1), comp_extra

        # # ------------------------
        # # 2) projections가 없을 때의 fallback mixture
        # #    - 각 채널에 ones-kernel conv 후 채널 평균
        # # ------------------------
        # N, C, H, W = x.shape
        
        # k1 = self._cast_buf(self.k_ones_f32, x)
        # act = F.conv2d(x, k1, stride=self.score_stride)  # (N,C,oh,ow)
        # act = act.mean(dim=1)  # (N,oh,ow)

        # comp_extra["proj_mixture"] = act
        # return act, comp_extra
    
        # 36채널을 먼저 1채널 평균으로 만든 뒤 1채널 전용 커널(k1)을 통과시킵니다.
        x_mean = x.mean(dim=1, keepdim=True) 
        k1 = self._cast_buf(self.k_ones_f32, x_mean)
        act = F.conv2d(x_mean, k1, stride=self.score_stride)  # (N,1,oh,ow)
        
        comp_extra["proj_mixture"] = act.squeeze(1) # (N,oh,ow)로 차원 축소
        return act.squeeze(1), comp_extra

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: (N, C, H, W)

        Returns
        -------
        score_map: (N, oh, ow)
        comp_maps: dict[str, (N, oh, ow)]
            - "density"
            - "centeredness"
            - "mixture"
            - "proj_A" (optional)
            - "proj_B" (optional)
            - "proj_mixture" (optional)
        """
        assert x.dim() == 4, f"KBRSConvScorer expects 4D tensor, got {x.shape}"

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
            comp_maps.update(extra)

        # score 합성
        score_map = torch.zeros_like(
            next(iter(comp_maps.values())), device=x.device, dtype=x.dtype
        )

        if "density" in comp_maps:
            score_map = score_map + self.weights.get("density", 0.0) * comp_maps["density"]
        if "centeredness" in comp_maps:
            score_map = score_map + self.weights.get("centeredness", 0.0) * comp_maps["centeredness"]
        if "mixture" in comp_maps:
            score_map = score_map + self.weights.get("mixture", 0.0) * comp_maps["mixture"]

        # comp_maps / score_map 수치 안정화
        for k, v in list(comp_maps.items()):
            v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            comp_maps[k] = v
            
        score_map = torch.nan_to_num(score_map, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 너무 커지지 않도록 한 번 클램프 (예: [-1e3, 1e3] 정도)
        score_map = score_map.clamp(-1e3, 1e3)
        
        return score_map, comp_maps


# ======================================================
# Wrapper: 기존 코드와의 호환을 위한 얇은 래퍼 (수정 버전)
# ======================================================
class KBRSWrapper(nn.Module):
    """
    예전 코드에서 사용하던 단순 인터페이스 래퍼.

    - region_size, score_weights, projections 를 받아서
      내부적으로 KBRSConvScorer 를 한 번만 생성해, 매 forward 에 재사용한다.
    """

    def __init__(
        self,
        region_size: Tuple[int, int] = (20, 12),
        score_weights: Optional[Dict[str, float]] = None,
        projections: Optional[Dict[str, List[int]]] = None,
        mixture_between: Optional[Tuple[str, str]] = None,  # (미사용, 시그니처 유지)
        mask_channel: Optional[int] = None,                 # (미사용, 시그니처 유지)
    ) -> None:
        super().__init__()
        self.kh, self.kw = region_size
        self.w: Dict[str, float] = dict(
            score_weights or {"density": 1.0, "mixture": 1.0, "centeredness": 1.0}
        )
        self.projections: Dict[str, List[int]] = projections or {}
        self.mixture_between = mixture_between
        self.mask_channel = mask_channel

        self.scorer = KBRSConvScorer(
            region_size=(self.kh, self.kw),
            weights=self.w,
            projections=self.projections,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 모델 전체에 .to(device) / .cuda() 를 걸면 self.scorer 도 같이 옮겨짐
        # dtype 은 내부에서 buffer 를 입력 x 에 맞춰 cast 하므로, 별도로 맞출 필요 없음.
        return self.scorer(x)
    