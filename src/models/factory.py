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
    build_maskrcnn_backbone,
)
from .plugins import KBRSHook
from .utils import pick_feature_map

class KBRSWrapper(nn.Module):
    """
    Attaches the KBRS auxiliary loss to a base model that has a `.backbone`.

    The score field is computed on the backbone's feature map, as in
    KBRS_MaskRCNN at e949efa, which trained the saliency-prior paper's models.
    A forward hook captures the backbone output during the base model's own
    forward pass, so the features are the ones the detection heads see (after
    the model's resize transform) and are not recomputed. An earlier version
    of this wrapper scored the raw input tensor instead, which has no trainable
    parameters upstream and so gave the loss no gradient.
    """
    def __init__(self, base_model: nn.Module, kbrs_params: dict = None, loss_weights: dict = None):
        super().__init__()
        self.base_model = base_model
        self.kbrs_hook = KBRSHook(kbrs_params=kbrs_params, loss_weights=loss_weights)
        backbone = getattr(base_model, "backbone", None)
        if backbone is None:
            raise ValueError("KBRSWrapper needs a base model with a .backbone attribute")
        self._captured = None
        backbone.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._captured = output

    def _feature_map(self) -> torch.Tensor:
        feats = self._captured
        self._captured = None
        if feats is None:
            raise RuntimeError("KBRSWrapper: the backbone did not run during the forward pass")
        if isinstance(feats, torch.Tensor):
            return feats
        _, fmap = pick_feature_map(feats, self.kbrs_hook.feature_map_name)
        return fmap

    def forward(self, images, targets=None):
        if self.training:
            base_out = self.base_model(images, targets)
            fmap = self._feature_map()
            _, loss_kbrs, comp = self.kbrs_hook(fmap, raw_images=images if isinstance(images, list) else None)

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
            out = self.base_model(images, targets)
            self._captured = None
            return out

def build_model(args: Any) -> nn.Module:
    """
    Build a model from a flat namespace of options (train.py and inference.py
    each assemble one), optionally wrapped with the KBRS auxiliary loss.

    model_name selects the architecture:
      maskrcnn            the proposal-detector observer (KBRS, ROCI and the
                          mode-target control are all built on it)
      director            Director-CenterNet, the ranked multi-region heatmap model
      centernet           plain CenterNet, the heatmap baseline
    RT-DETR, Deformable DETR and the video DETR variants were moved to
    archive/legacy/ in 2026-10.
    """
    model_name = getattr(args, "model_name", "centernet").lower()
    use_kbrs = getattr(args, "use_kbrs", False)
    num_classes = getattr(args, "num_classes", 2)
    in_channels = getattr(args, "in_channels", 36)
    window_size = getattr(args, "window_size", 4)
    loss_weights = getattr(args, "loss_weights", {})
    kbrs_params = getattr(args, "kbrs_params", {}) or {}

    use_dp = getattr(args, "use_density_peak", False) or (model_name == "centernet_density_peak")

    if model_name == "maskrcnn":
        model = build_maskrcnn_backbone(
            num_classes=num_classes,
            window_size=window_size,
            in_channels=in_channels,
            # Neither train.py nor inference.py sets resize_mode, so every
            # Mask R-CNN built here (all v6 runs included) resizes its input to 640x640,
            # whatever conf/architecture/maskrcnn.yaml says. Changing this
            # default changes the model; keep it for comparability.
            resize_mode=getattr(args, "resize_mode", "fixed_square"),
            do_normalize=getattr(args, "do_normalize", False),
            soft_mode_cls=getattr(args, "soft_mode_cls", False),
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
            smooth_huber_delta=getattr(args, "smooth_huber_delta", config.DIRECTOR_SMOOTH_HUBER_DELTA),
            smooth_warmup_start=getattr(args, "smooth_warmup_start", config.DIRECTOR_SMOOTH_WARMUP_START),
            smooth_warmup_full=getattr(args, "smooth_warmup_full", config.DIRECTOR_SMOOTH_WARMUP_FULL),
            soft_center_radius=getattr(args, "soft_center_radius", config.DIRECTOR_SOFT_CENTER_RADIUS),
            peak_border_margin=getattr(args, "peak_border_margin", config.DIRECTOR_PEAK_BORDER_MARGIN),
            trainable_layers=getattr(args, "trainable_layers", config.DIRECTOR_TRAINABLE_LAYERS),
            head_conv=getattr(args, "head_conv", config.DIRECTOR_HEAD_CONV),
            dense_positives=getattr(args, "dense_positives", config.DIRECTOR_DENSE_POSITIVES),
            hcm_negative_target=getattr(args, "hcm_negative_target", config.DIRECTOR_HCM_NEGATIVE_TARGET),
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}. Supported: 'maskrcnn', 'director', 'centernet' "
                         f"(the DETR variants are in archive/legacy/)")

    if use_kbrs:
        kbrs_params = dict(kbrs_params or {})
        kbrs_params.setdefault("window_size", window_size)
        kbrs_params.setdefault("per_window", max(1, in_channels // max(1, window_size)))
        model = KBRSWrapper(base_model=model, kbrs_params=kbrs_params, loss_weights=loss_weights)

    if model_name.startswith("centernet"):
        Logger.info(f"[Model] density peak plugin: {'on' if use_dp else 'off'}")
    for key, value in describe_model(model).items():
        Logger.info(f"[Model] {key}: {value}")
    return model


def describe_model(model: nn.Module) -> dict:
    """The settings the built model actually uses, read off the model itself.

    Read back rather than taken from the config, because several config
    values never reach the model (the KBRS weight is always 0.25, Mask R-CNN
    always resizes to 640). train.py logs this at start-up and stores it in
    the W&B config under "effective".
    """
    base = model.base_model if isinstance(model, KBRSWrapper) else model
    out = {"architecture": type(base).__name__}

    in_conv = getattr(getattr(getattr(base, "backbone", None), "body", None), "conv1", None)
    if in_conv is not None:
        out["input_channels"] = in_conv.in_channels

    transform = getattr(base, "transform", None)  # Mask R-CNN
    if transform is not None and hasattr(transform, "min_size"):
        out["input_size"] = f"min_size {tuple(transform.min_size)}, max_size {transform.max_size}"
    roi_heads = getattr(base, "roi_heads", None)
    if roi_heads is not None:
        out["soft_mode_cls"] = type(roi_heads).__name__ != "RoIHeads"

    if hasattr(base, "hcm_negative_target"):  # Director-CenterNet
        out["loss_weights"] = dict(base.loss_weights)
        out["hcm_negative_target"] = base.hcm_negative_target
        out["down_ratio"] = base.down_ratio
        out["head_conv"] = base.head_conv
        out["imagenet_backbone"] = getattr(base, "imagenet_weights_loaded", None)

    if isinstance(model, KBRSWrapper):
        hook = model.kbrs_hook
        out["kbrs"] = {
            "loss_weight": float(hook.loss_weights.get("loss_kbrs", 0.25)),
            "component_weights": dict(hook.weights),
            "region_size": list(hook.region_size),
            "tau": hook.tau,
            "gate_channels": len(hook.gate_channels),
            "mixture_nonneg": hook.kbrs_params.get("mixture_nonneg"),
            "feature_map": hook.feature_map_name,
        }
    else:
        out["kbrs"] = "off"
    return out