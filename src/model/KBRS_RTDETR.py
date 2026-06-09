import torch
from .RTDETR import BaseRTDETR
from .kbrs import KBRSConvScorer
from .utils import normalize_projections, auto_expand_indices, compute_gate_from_raw_inputs, aux_boost_loss

# ==========================================
# 자식 클래스 2: KBRS가 결합된 RT-DETR
# ==========================================
class KBRS_RTDETR(BaseRTDETR):
    def __init__(self, weights="rtdetrv2-l.pt", num_classes=2, in_channels=3, kbrs_params=None, loss_weights=None):
        # 1. 부모 클래스 초기화 (모델 로드, 개조, 손실함수 세팅)
        super().__init__(weights=weights, num_classes=num_classes, in_channels=in_channels)
        
        self.kbrs_params = kbrs_params or {}
        self.loss_weights = dict(loss_weights or {})
        
        # 2. KBRS Scorer 변수 세팅 (Hook 기능 완벽 삭제!)
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
        
        # 3. KBRS 모듈 생성
        in_channels_total = self._per_window * self._window_size
        proj_norm = normalize_projections(self.kbrs_params.get("projections", None), self._window_size, self._per_window, in_channels_total)
        
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
            
            # =========================================================
            # [핵심 수정] 백본 피처맵이 아닌 원본 'batched_images'를 KBRS에 직행하되,
            # KBRS 내부의 필터가 1채널을 기대하므로 채널을 합쳐서(mean) 넘겨줍니다.
            # (이 방식은 밀도(Density) 계산을 위한 가장 범용적인 우회로입니다.)
            # =========================================================
            fmap = batched_images.detach() if self._detach_scorer_input else batched_images
            
            # [추가된 부분] 36채널을 1채널로 압축하여 k1 필터(1채널)와 호환되게 만듭니다.
            # 만약 kbrs.py 내부에서 36채널을 직접 슬라이싱하는 로직이 완벽히 구현되어 있다면 
            # 이 줄을 빼야 하지만, 현재 에러가 나므로 이 방법으로 강제 호환시킵니다.
            fmap_for_kbrs = fmap.mean(dim=1, keepdim=True) 

            self.kbrs_scorer = self.kbrs_scorer.to(fmap_for_kbrs.device, dtype=fmap_for_kbrs.dtype)
            
            # 압축된 1채널 맵을 넘깁니다.
            score_map, _ = self.kbrs_scorer(fmap_for_kbrs)
            
            # Gate 연산 (여기서는 원본 36채널 raw_images를 그대로 사용)
            in_channels_total = self._per_window * self._window_size
            gate_channels = auto_expand_indices(self._gate_channels_cfg, self._window_size, self._per_window, in_channels_total)
            
            if len(gate_channels) > 0:
                gain = compute_gate_from_raw_inputs(
                    raw_images, score_map.unsqueeze(1), gate_channels,
                    self._scorer_region_size, 1, self._gate_gain
                )
                score_map = score_map * gain
                
            # 3) KBRS 손실 계산 및 최종 취합
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