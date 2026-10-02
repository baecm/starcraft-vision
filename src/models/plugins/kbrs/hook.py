# src/model/plugins/kbrs/hook.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn

from .feature_scorer import KBRSFeatureScorer
from ...utils import normalize_projections, auto_expand_indices, compute_gate_from_raw_inputs, aux_boost_loss


class KBRSHook(nn.Module):
    """
    Computes the KBRS score field on a detector feature map and its auxiliary
    loss, in the order KBRS_MaskRCNN.forward used at e949efa: score the
    (optionally detached) feature map, gate the score by the visibility
    channels of the raw input, then apply the contrast loss with temperature
    kbrs_params["tau"].

    Two departures from the August-refactor version of this file, both to
    restore what trained the paper's models: the gate channels are expanded
    over the stacked window like the projections are, and the loss uses
    "tau" (4.0) rather than the scorer's unused "mixture_tau" (2.0).
    """

    def __init__(self, kbrs_params: Optional[Dict[str, Any]] = None, loss_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.kbrs_params = dict(kbrs_params or {})
        self.loss_weights = dict(loss_weights or {})

        self.window_size = int(self.kbrs_params.get("window_size", 4))
        self.per_window = int(self.kbrs_params.get("per_window", 9))
        in_channels_total = self.per_window * self.window_size

        # The component weights live under "score_weights" in older configs and
        # flat in newer ones; a run that read neither silently trained at 1/1/1.
        sw = dict(self.kbrs_params.get("score_weights") or {})
        self.weights = {
            k: float(self.kbrs_params[k]) if self.kbrs_params.get(k) is not None else float(sw.get(k, 1.0))
            for k in ("density", "mixture", "centeredness")
        }

        self.region_size = tuple(self.kbrs_params.get("region_size", (20, 12)))
        self.gate_channels = auto_expand_indices(
            list(self.kbrs_params.get("gate_channels", [])), self.window_size, self.per_window, in_channels_total
        )
        self.gate_gain = float(self.kbrs_params.get("gate_gain", 1.0))
        self.gate_reduce = str(self.kbrs_params.get("gate_reduce", "max")).lower()
        self.detach_scorer_input = bool(self.kbrs_params.get("detach_scorer_input", False))
        self.tau = float(self.kbrs_params.get("tau", 4.0))
        self.feature_map_name = str(self.kbrs_params.get("feature_map_name", "smallest"))

        proj_norm = normalize_projections(
            self.kbrs_params.get("projections", None), self.window_size, self.per_window, in_channels_total
        )
        self.scorer = KBRSFeatureScorer(
            region_size=self.region_size,
            weights=self.weights,
            projections=proj_norm,
            mixture_mode=str(self.kbrs_params.get("mixture_mode", "confusion")),
            mixture_power=float(self.kbrs_params.get("mixture_power", 1.0)),
            score_stride=int(self.kbrs_params.get("score_stride", 1)),
            downsample_before=self.kbrs_params.get("downsample_before", None),
        )
        # Last components, kept so a smoke test or a logger can inspect them.
        self.last_components: Dict[str, torch.Tensor] = {}

    def forward(self, feature_map: torch.Tensor, raw_images: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        fmap = feature_map.detach() if self.detach_scorer_input else feature_map
        score_map, components = self.scorer(fmap)

        if raw_images is not None and len(self.gate_channels) > 0:
            gate_channels = [c for c in self.gate_channels if c < raw_images[0].shape[0]]
            gate = compute_gate_from_raw_inputs(
                raw_images=raw_images,
                ref_fmap=score_map,
                gate_channels=gate_channels,
                region_size=self.region_size,
                score_stride=self.scorer.score_stride,
                gate_gain=self.gate_gain,
                reduce=self.gate_reduce,
            )
            score_map = score_map * gate
            components["gate_gain"] = gate

        loss_kbrs = aux_boost_loss(score_map, tau=self.tau)
        self.last_components = {k: v.detach() for k, v in components.items()}
        return score_map, loss_kbrs, components
