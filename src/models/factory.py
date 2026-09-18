# src/model/factory.py
from __future__ import annotations
import os
import torch
import torch.nn as nn
from typing import Any

try:
    from utils.logger import Logger
except ModuleNotFoundError:
    from src.utils.logger import Logger
try:
    import config
except ModuleNotFoundError:
    from src import config
from .backbones import (
    CenterNetBackbone,
    DirectorCenterNet,
    DeformableDETRBackbone,
    ProbabilisticVideoDETR,
    build_maskrcnn_backbone,
    build_rtdetr_backbone,
)
from .plugins import KBRSHook

class KBRSWrapper(nn.Module):
    """
    Generic KBRS Model Wrapper that attaches KBRSHook to any base backbone model.
    """
    def __init__(self, base_model: nn.Module, kbrs_params: dict = None, loss_weights: dict = None):
        super().__init__()
        self.base_model = base_model
        self.kbrs_hook = KBRSHook(kbrs_params=kbrs_params, loss_weights=loss_weights)

    def forward(self, images, targets=None):
        if self.training:
            base_out = self.base_model(images, targets)
            batched_images = torch.stack(images, dim=0) if isinstance(images, list) else images
            _, loss_kbrs, comp = self.kbrs_hook(batched_images, raw_images=images if isinstance(images, list) else None)

            losses = {}
            if isinstance(base_out, dict):
                losses.update(base_out)
            else:
                losses["loss_base"] = base_out

            # engine_safe backprops sum(loss_dict.values()); adding an aggregate
            # here re-counted every base loss on top of the base model's own.
            w_kbrs = self.kbrs_hook.loss_weights.get("loss_kbrs", 0.25)
            losses["loss_kbrs"] = w_kbrs * loss_kbrs
            return losses
        else:
            return self.base_model(images, targets)

def build_model(args: Any) -> nn.Module:
    """
    Unified Factory Entrypoint to build object detection models with optional plugins (KBRS, Density Peak, Probabilistic Query).
    """
    model_name = getattr(args, "model_name", "centernet").lower()
    use_kbrs = getattr(args, "use_kbrs", False)
    num_classes = getattr(args, "num_classes", 2)
    in_channels = getattr(args, "in_channels", 36)
    window_size = getattr(args, "window_size", 4)
    loss_weights = getattr(args, "loss_weights", {})
    kbrs_params = getattr(args, "kbrs_params", {}) or {}

    use_dp = getattr(args, "use_density_peak", False) or (model_name == "centernet_density_peak")
    use_pq = getattr(args, "use_probabilistic_query", False) or ("probabilistic" in model_name)

    # ------------------------------------------------------------------
    # Model Construction
    # ------------------------------------------------------------------
    if model_name == "maskrcnn":
        model = build_maskrcnn_backbone(
            num_classes=num_classes,
            window_size=window_size,
            in_channels=in_channels,
            resize_mode=getattr(args, "resize_mode", "fixed_square"),
            do_normalize=getattr(args, "do_normalize", False)
        )

    elif model_name == "rtdetr":
        model = build_rtdetr_backbone(
            num_classes=num_classes,
            version=getattr(args, "rtdetr_version", "v1"),
            model_size=getattr(args, "rtdetr_size", "l"),
            in_channels=in_channels
        )

    elif model_name in ["centernet", "centernet_density_peak"]:
        model = CenterNetBackbone(
            num_classes=num_classes,
            in_channels=in_channels,
            down_ratio=getattr(args, "centernet_down_ratio", 4),
            max_objs=getattr(args, "max_objs", 100),
            use_density_peak=use_dp,
            loss_weights=loss_weights
        )

    elif model_name in ["director_centernet", "director-centernet", "director_centernet_win4", "director"]:
        model = DirectorCenterNet(
            in_channels=in_channels,
            num_classes=1,
            down_ratio=getattr(args, "centernet_down_ratio", 4),
            k_max=getattr(args, "k_max", getattr(config, "DIRECTOR_K", 3)),
            conf_threshold=getattr(args, "conf_threshold", getattr(config, "DIRECTOR_TAU", 0.2)),
            render_sigma=getattr(args, "render_sigma", getattr(config, "DIRECTOR_RENDER_SIGMA", 2.0)),
            u_observers=getattr(args, "u_observers", getattr(config, "NUM_OBSERVERS_U", 5)),
            viewport_size_hw=getattr(args, "viewport_size_hw", getattr(config, "VIEWPORT_SIZE_HW", (12, 20))),
            loss_weights=loss_weights,
        )

    elif model_name in ["probabilistic_video_detr", "video_detr", "deformable_video_detr_cvae"]:
        per_frame_c = getattr(args, "single_frame_channels", None)
        if per_frame_c is None:
            per_frame_c = in_channels // max(1, window_size) if in_channels > window_size else in_channels
        model = ProbabilisticVideoDETR(
            in_channels=per_frame_c,
            feat_dim=getattr(args, "feat_dim", 128),
            num_raters=getattr(args, "num_raters", 5),
            latent_dim=getattr(args, "latent_dim", 64),
            num_queries=getattr(args, "num_queries", 3),
            num_heads=getattr(args, "num_heads", 8),
            num_decoder_layers=getattr(args, "num_decoder_layers", 3),
            grid_size=getattr(args, "grid_size", (128, 128)),
            use_cvae=getattr(args, "use_cvae", True),
        )

    elif model_name in ["deformable_detr", "deformable_video_detr", "deformable_video_detr_probabilistic"]:
        model = DeformableDETRBackbone(
            num_classes=num_classes,
            in_channels=in_channels,
            window_size=window_size,
            num_queries=getattr(args, "num_queries", 100),
            use_probabilistic_query=use_pq,
            loss_weights=loss_weights
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}. Supported: 'maskrcnn', 'rtdetr', 'centernet', 'deformable_detr', 'probabilistic_video_detr'")

    if use_kbrs:
        kbrs_params = dict(kbrs_params or {})
        kbrs_params.setdefault("window_size", window_size)
        kbrs_params.setdefault("per_window", max(1, in_channels // max(1, window_size)))
        model = KBRSWrapper(base_model=model, kbrs_params=kbrs_params, loss_weights=loss_weights)

    # ------------------------------------------------------------------
    # Structured Model & Plugin Logging
    # ------------------------------------------------------------------
    log_lines = [
        "======================================================================",
        f"[MODEL BUILD] Architecture : {model_name.upper()}",
        f"[MODEL BUILD] Input Chans  : {in_channels} (Window Size: {window_size})",
        "----------------------------------------------------------------------",
        "[PLUGINS ATTACHED SUMMARY]",
        f"  - KBRS Plugin          : {'[ENABLED]' if use_kbrs else '[DISABLED]'}",
        f"  - Density Peak Plugin  : {'[ENABLED]' if (model_name.startswith('centernet') and use_dp) else ('[DISABLED]' if model_name.startswith('centernet') else '[N/A]')}",
        f"  - Probabilistic Query  : {'[ENABLED]' if ('detr' in model_name and use_pq) else ('[DISABLED]' if 'detr' in model_name else '[N/A]')}",
        "======================================================================"
    ]
    for line in log_lines:
        try:
            Logger.info(line)
        except Exception:
            print(line)

    return model