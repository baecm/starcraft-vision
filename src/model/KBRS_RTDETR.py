import torch
from .CustomRTDETR import BaseRTDETR
from .KBRSConvScorer import KBRSConvScorer
from .utils import normalize_projections, auto_expand_indices, compute_gate_from_raw_inputs, aux_boost_loss

# ==========================================
# 자식 클래스 2: KBRS가 결합된 RT-DETR
# ==========================================
class KBRS_RTDETR(BaseRTDETR):
    def __init__(self, weights="rtdetrv2-l.pt", num_classes=2, in_channels=3, kbrs_params=None, loss_weights=None):
        super().__init__(weights=weights, num_classes=num_classes, in_channels=in_channels)
        
        self.kbrs_params = kbrs_params or {}
        self.loss_weights = dict(loss_weights or {})
        
        # KBRS Scorer 변수 세팅
        self._window_size = int(self.kbrs_params.get("window_size", 1))
        self._per_window  = int(self.kbrs_params.get("per_window", 9))
        self._weights = {
            "density": self.kbrs_params.get("density", 1.0),
            "mixture": self.kbrs_params.get("mixture", 1.0),
            "centeredness": self.kbrs_params.get("centeredness", 1.0),
        }
        self._scorer_region_size = tuple(self.kbrs_params.get("region_size", (20, 12)))
        self._gate_channels_cfg = list(self.kbrs_params.get("gate_channels", []))
        self._gate_gain = float(self.kbrs_params.get("gate_gain", 1.0))
        self._detach_scorer_input = bool(self.kbrs_params.get("detach_scorer_input", True))
        
        in_channels_total = self._per_window * self._window_size
        proj_norm = normalize_projections(self.kbrs_params.get("projections", None), self._window_size, self._per_window, in_channels_total)
        print(f"[KBRS_RTDETR] Initialized with KBRS parameters: {self.kbrs_params}")
        print(f"[KBRS_RTDETR] Normalized Projections: {proj_norm}")
        
        self.kbrs_scorer = KBRSConvScorer(
            region_size=self._scorer_region_size,
            weights=self._weights,
            projections=proj_norm,
            mixture_tau=float(self.kbrs_params.get("mixture_tau", 2.0)),
            mixture_mode=str(self.kbrs_params.get("mixture_mode", "confusion")),
            mixture_power=float(self.kbrs_params.get("mixture_power", 1.0)),
            score_stride=int(self.kbrs_params.get("score_stride", 1)),
            downsample_before=self.kbrs_params.get("downsample_before", None),
        )

    def forward(self, images, targets=None):
        raw_images = images
        batched_images = torch.stack(images, dim=0) if isinstance(images, list) else images
            
        if self.training:
            if targets is None: raise ValueError("In training mode, targets should be passed")
            
            # 1) 원시 네트워크 순전파 및 RT-DETR 기본 손실 계산
            batch_dict = self._format_targets(targets, batched_images)
            preds = self.model._predict_once(batched_images)
            preds_for_loss = preds[:2] if isinstance(preds, tuple) and len(preds) > 2 else preds
                
            criterion_out = self.criterion(preds_for_loss, batch_dict)
            
            losses = {}
            if isinstance(criterion_out, dict): losses.update(criterion_out)
            elif isinstance(criterion_out, tuple): losses["loss_rtdetr"] = criterion_out[0]
            else: losses["loss_rtdetr"] = criterion_out
            
            fmap_for_kbrs = batched_images.detach() if self._detach_scorer_input else batched_images            
            
            self.kbrs_scorer = self.kbrs_scorer.to(fmap_for_kbrs.device, dtype=fmap_for_kbrs.dtype)
            
            score_map, _ = self.kbrs_scorer(fmap_for_kbrs)
            
            in_channels_total = self._per_window * self._window_size
            gate_channels = auto_expand_indices(self._gate_channels_cfg, self._window_size, self._per_window, in_channels_total)
            
            if len(gate_channels) > 0:
                gain = compute_gate_from_raw_inputs(
                    raw_images, score_map.unsqueeze(1), gate_channels,
                    self._scorer_region_size, 1, self._gate_gain
                )
                score_map = score_map * gain
                
            tau = float(self.kbrs_params.get("tau", 2.0))
            losses["loss_kbrs"] = aux_boost_loss(score_map, tau=tau)
            
            for k in list(losses.keys()):
                w = float(self.loss_weights.get(k, 1.0))
                losses[k] = losses[k] * w
                
            return losses
            
        else:
            with torch.no_grad():
                preds = self.model(batched_images)
            return self._format_predictions(preds, batched_images.shape[-2:])