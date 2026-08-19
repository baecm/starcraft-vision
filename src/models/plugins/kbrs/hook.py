# src/model/plugins/kbrs/hook.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn

from .scorer import KBRSConvScorer
from ...utils import normalize_projections, compute_gate_from_raw_inputs, aux_boost_loss

class KBRSHook(nn.Module):
    """
    Unified KBRS Hook Wrapper module that attaches to feature maps
    and calculates KBRS score maps and auxiliary losses.
    """
    def __init__(self, kbrs_params: Optional[Dict[str, Any]] = None, loss_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.kbrs_params = kbrs_params or {}
        self.loss_weights = dict(loss_weights or {})

        self.window_size = int(self.kbrs_params.get("window_size", 4))
        self.per_window = int(self.kbrs_params.get("per_window", 9))
        self.weights = {
            "density": self.kbrs_params.get("density", 1.0),
            "mixture": self.kbrs_params.get("mixture", 1.0),
            "centeredness": self.kbrs_params.get("centeredness", 1.0),
        }
        self.region_size = tuple(self.kbrs_params.get("region_size", (20, 12)))
        self.gate_channels = list(self.kbrs_params.get("gate_channels", []))
        self.gate_gain = float(self.kbrs_params.get("gate_gain", 1.0))
        self.gate_reduce = str(self.kbrs_params.get("gate_reduce", "max")).lower()
        self.detach_scorer_input = bool(self.kbrs_params.get("detach_scorer_input", True))

        in_channels_total = self.per_window * self.window_size
        proj_norm = normalize_projections(
            self.kbrs_params.get("projections", None),
            self.window_size,
            self.per_window,
            in_channels_total
        )

        self.scorer = KBRSConvScorer(
            region_size=self.region_size,
            weights=self.weights,
            projections=proj_norm,
            mixture_tau=float(self.kbrs_params.get("mixture_tau", 2.0)),
            mixture_mode=str(self.kbrs_params.get("mixture_mode", "confusion")),
            mixture_power=float(self.kbrs_params.get("mixture_power", 1.0)),
            score_stride=int(self.kbrs_params.get("score_stride", 1)),
            downsample_before=self.kbrs_params.get("downsample_before", None),
        )

    def forward(self, feature_map: torch.Tensor, raw_images: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        fmap = feature_map.detach() if self.detach_scorer_input else feature_map
        self.scorer = self.scorer.to(fmap.device, dtype=fmap.dtype)

        score_map, components = self.scorer(fmap)

        if raw_images is not None:
            gate_channels_norm = [c for c in self.gate_channels if c < raw_images[0].shape[0]]
            gate = compute_gate_from_raw_inputs(
                raw_images=raw_images,
                ref_fmap=score_map,
                gate_channels=gate_channels_norm,
                region_size=self.region_size,
                score_stride=self.scorer.score_stride,
                gate_gain=self.gate_gain,
                reduce=self.gate_reduce,
            )
            gated_score_map = score_map * gate
        else:
            gated_score_map = score_map

        loss_kbrs = aux_boost_loss(gated_score_map, tau=self.scorer.mixture_tau)
        return gated_score_map, loss_kbrs, components
