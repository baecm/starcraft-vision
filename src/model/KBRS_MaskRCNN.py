# src/model/KBRS_MaskRCNN.py
import torch
import torch.nn as nn

from torchvision.models.detection import MaskRCNN

from collections import OrderedDict
from typing import Dict, List

from .kbrs import KBRSConvScorer
from .utils import pick_feature_map, normalize_projections, auto_expand_indices, compute_gate_from_raw_inputs, reduce_map_stats, aux_boost_loss, aux_entropy_sharp


class KBRS_MaskRCNN(MaskRCNN):
    def __init__(self, backbone, num_classes,
                 kbrs_params=None, loss_weights=None):
        super().__init__(backbone, num_classes)
        self.kbrs_params = kbrs_params or {}

        # ===== Loss weights (최종 합산 비중) =====
        self.loss_weights = dict(loss_weights or {})        # e.g., {'loss_kbrs':0.25, 'loss_objectness':1.0, ...}
        self.learnable_loss_weights = self.kbrs_params.get("learnable", None)  # 'static'이면 비활성
        self._loss_weight_head = None

        # runtime logging
        self.log_into_losses = bool(self.kbrs_params.get("log_into_losses", False))
        self.kbrs_last_logs = {}
        self.kbrs_cache = None

        # ===== Scorer / KBRS params =====
        self._scorer_impl       = self.kbrs_params.get('scorer_impl', 'conv')
        self._scorer_region_size= tuple(self.kbrs_params.get("region_size", (20, 12)))
        self._scorer_stride     = int(self.kbrs_params.get("score_stride", 1))
        self._mixture_tau       = float(self.kbrs_params.get("mixture_tau", 2.0))
        self._mixture_mode      = str(self.kbrs_params.get("mixture_mode", "confusion"))
        self._mixture_power     = float(self.kbrs_params.get("mixture_power", 1.0))

        self._gate_channels_cfg: List[int] = list(self.kbrs_params.get("gate_channels", []))
        self._gate_reduce: str  = str(self.kbrs_params.get("gate_reduce", "max")).lower()
        self._gate_gain: float  = float(self.kbrs_params.get("gate_gain", self.kbrs_params.get("mask_gain", 1.0)))
        self._detach_scorer_input: bool = bool(self.kbrs_params.get("detach_scorer_input", True))

        self._projections_cfg = self.kbrs_params.get("projections", None)
        # score_weights: 내부 합성 비율 (구식 'weights'도 허용)
        _ws = self.kbrs_params.get("score_weights", None)
        if _ws is None:
            _ws = self.kbrs_params.get("weights", None)  # backward-compat
        self._weights = dict(_ws or {"density": 1.0, "mixture": 1.0, "centeredness": 1.0})

        self._mixture_between = self.kbrs_params.get("mixture_between", None)  # (미사용)
        self._downsample_before = self.kbrs_params.get("downsample_before", None)

        self._viz_components = bool(self.kbrs_params.get("viz_components", True))
        self._acc_epoch = bool(self.kbrs_params.get("accumulate_epoch", True))

        # window-size 메타 (builder가 주입)
        self._window_size = int(self.kbrs_params.get("window_size", 1))
        self._per_window  = int(self.kbrs_params.get("per_window", 9))

        # scorer placeholder
        self.kbrs_scorer = None

    def _maybe_init_loss_head(self, loss_keys, device):
        if self.learnable_loss_weights and self.learnable_loss_weights != "static" and self._loss_weight_head is None:
            self._loss_weight_head = nn.ParameterDict({
                k: nn.Parameter(torch.tensor(1.0, device=device))
                for k in loss_keys
            })

    def consume_epoch_kbrs(self):
        """train.py에서 epoch끝에 불러 WANDB로 올릴 수 있게 제공."""
        scalars = None
        cache = None
        if hasattr(self, "_kbrs_epoch_sums") and self._kbrs_epoch_count > 0:
            cnt = float(self._kbrs_epoch_count)
            scalars = {f"kbrs_epoch/{k}": v / cnt for k, v in self._kbrs_epoch_sums.items()}
            self._kbrs_epoch_sums = {}
            self._kbrs_epoch_count = 0
        if self.kbrs_cache is not None:
            cache = self.kbrs_cache
            self.kbrs_cache = None
        return scalars, cache

    def forward(self, images, targets=None):
        """
        Full forward (train/eval):
        1) transform
        2) backbone(FPN)
        3) pick FPN feature & compute KBRS score_map
        4) (optional) gate with vision-like channels (from raw inputs)
        5) RPN -> proposals
        6) ROI heads -> detections
        7) postprocess or assemble losses (+ KBRS losses)
        8) (optional) apply (learnable) loss weights
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # 1) keep raw inputs for gate, then transform
        raw_images = images  # list[Tensor (Cin,H,W)]
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)

        # 2) backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # 3) pick feature map
        fmap_key_pref = self.kbrs_params.get("feature_map_name", "smallest")
        fmap_key, fmap = pick_feature_map(features, fmap_key_pref)

        # projections normalize & scorer init (once)
        in_channels_total = self._per_window * self._window_size
        proj_norm = normalize_projections(
            self._projections_cfg, self._window_size, self._per_window, in_channels_total
        )

        if self.kbrs_scorer is None:
            self.kbrs_scorer = KBRSConvScorer(
                region_size=self._scorer_region_size,
                weights=self._weights,
                projections=proj_norm,
                mixture_tau=self._mixture_tau,
                mixture_mode=self._mixture_mode,
                mixture_power=self._mixture_power,
                mask_channel=None,
                mask_gain=1.0,
                score_stride=self._scorer_stride,
                downsample_before=self._downsample_before,
            )
        self.kbrs_scorer = self.kbrs_scorer.to(fmap.device, dtype=fmap.dtype)

        # scorer 입력 detach 옵션
        fmap_for_kbrs = fmap.detach() if self._detach_scorer_input else fmap

        # KBRS score map
        score_map, comp_maps = self.kbrs_scorer(fmap_for_kbrs)

        # 4) Vision-like 게이트 (원본 입력 기반)
        gate_channels = auto_expand_indices(
            self._gate_channels_cfg, self._window_size, self._per_window, in_channels_total
        )
        if len(gate_channels) > 0:
            gain = compute_gate_from_raw_inputs(
                raw_images, score_map.unsqueeze(1), gate_channels,
                self._scorer_region_size, self._scorer_stride, self._gate_gain, reduce=self._gate_reduce
            )  # (N,oh,ow)
            score_map = score_map * gain
            comp_maps["gate_gain"] = gain

        # === per-component statistics (scalar) ===
        want_keys = ["density", "mixture", "centeredness", "proj_A", "proj_B", "proj_mixture", "gate_gain"]
        logs = {}
        for k in want_keys:
            if k in comp_maps:
                stats = reduce_map_stats(comp_maps[k].detach())
                for stat_name, vec in stats.items():
                    logs[f"{k}/{stat_name}"] = vec
        scalar_logs = {f"{name}_meanB": val.mean() for name, val in logs.items()}
        self.kbrs_last_logs = scalar_logs  # epoch 누적에 사용

        # --- cache one sample (1장) for WANDB viz ---
        if self._viz_components and self.training:
            with torch.no_grad():
                self.kbrs_cache = {
                    "score_total": score_map[:1].detach().cpu(),
                    "comp_maps":   {k: v[:1].detach().cpu() for k, v in comp_maps.items()},
                }

        # --- accumulate epoch scalars ---
        if self._acc_epoch and self.training:
            if not hasattr(self, "_kbrs_epoch_sums"):
                self._kbrs_epoch_sums = {}
                self._kbrs_epoch_count = 0
            if self.kbrs_last_logs:
                for k, v in self.kbrs_last_logs.items():
                    self._kbrs_epoch_sums[k] = self._kbrs_epoch_sums.get(k, 0.0) + float(v)
                self._kbrs_epoch_count += 1

        # 5) RPN
        proposals, proposal_losses = self.rpn(images, features, targets)
        # 6) ROI heads
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # 7) postprocess (eval)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        if not self.training:
            return detections

        # ===== Assemble losses (train) =====
        losses: Dict[str, torch.Tensor] = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # --- KBRS: 메인 손실만 최적화 반영 ---
        tau = float(self.kbrs_params.get("tau", 2.0))
        losses["loss_kbrs"] = aux_boost_loss(score_map, tau=tau)

        # (옵션) 엔트로피 항은 최적화 반영 여부를 설정으로 제어
        if bool(self.kbrs_params.get("use_entropy", False)):
            losses["loss_kbrs_entropy"] = aux_entropy_sharp(score_map)

        # --- 컴포넌트 손실은 '로그 전용'으로만 계산 (총손실 미포함) ---
        if bool(self.kbrs_params.get("component_losses", True)):
            per_tau = {"density": 0.5, "mixture": 1.2, "centeredness": 0.5}
            with torch.no_grad():
                for ck in ["density", "mixture", "centeredness"]:
                    if ck in comp_maps:
                        val = aux_boost_loss(
                            comp_maps[ck], tau=per_tau.get(ck, 1.0), norm="zscore"
                        )
                        self.kbrs_last_logs[f"comp_loss/{ck}"] = float(val.detach())

        # --- apply weights (learnable or static) ---
        self._maybe_init_loss_head(losses.keys(), images.tensors.device)
        if self._loss_weight_head is not None:
            w = {k: self._loss_weight_head[k] for k in losses.keys()}
            for k in list(losses.keys()):
                losses[k] = losses[k] * w.get(k, torch.tensor(1.0, device=images.tensors.device))
        else:
            for k in list(losses.keys()):
                default_w = 1.0
                losses[k] = losses[k] * float(self.loss_weights.get(k, default_w))

        return losses
