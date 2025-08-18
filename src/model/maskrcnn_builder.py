# maskrcnn_builder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

from .CustomRCNNTransform import CustomRCNNTransform
from .kbrs import KBRSConvScorer, KBRSUnfoldScorer


# ============================== Utils ==============================

def _reduce_map_stats(comp_map: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    comp_map: (N, oh, ow)
    return: {'mean': (N,), 'max': (N,), 'sum': (N,)}
    """
    N = comp_map.size(0)
    flat = comp_map.view(N, -1)
    return {
        "mean": flat.mean(dim=1),
        "max":  flat.max(dim=1).values,
        "sum":  flat.sum(dim=1),
    }


def aux_boost_loss(score_map, tau=2.0):
    return -torch.mean(torch.log(1e-6 + torch.sigmoid(score_map * tau)))


def aux_entropy_sharp(score_map):
    p = torch.sigmoid(score_map)
    return -torch.mean(p * torch.log(p + 1e-6))


def _auto_expand_indices(indices: List[int], window_size: int, per_window: int, in_channels: int) -> List[int]:
    """
    indices가 per_window(기본 9) 기준의 '단일 윈도우' 인덱스일 때 window_size 배로 자동 확장.
    예) indices=[8], window_size=4, per_window=9 -> [8,17,26,35]
    조건:
      - window_size > 1
      - max(indices) < per_window
      - in_channels == per_window * window_size
    """
    if not indices:
        return indices
    if window_size > 1 and max(indices) < per_window and in_channels == per_window * window_size:
        out = []
        for base in indices:
            out.extend([base + t * per_window for t in range(window_size)])
        return sorted(list(set(out)))
    return indices


def _normalize_projections(
    projections_cfg,
    window_size: int,
    per_window: int,
    in_channels: int
) -> Dict[str, List[int]]:
    """
    projections가 [{'name':'A','channels':[0,1,2,3]}, ...] 또는 {'A':[...], 'B':[...]} 형태로 올 수 있음.
    모두 dict(name->List[int])로 통일하고, 필요 시 window-size에 맞춰 채널 자동 확장.
    """
    if projections_cfg is None:
        return {}
    proj_dict: Dict[str, List[int]] = {}

    if isinstance(projections_cfg, dict):
        for name, chs in projections_cfg.items():
            proj_dict[name] = _auto_expand_indices(list(chs), window_size, per_window, in_channels)
        return proj_dict

    if isinstance(projections_cfg, list):
        for item in projections_cfg:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            chs  = item.get("channels", [])
            if name is None:
                continue
            proj_dict[name] = _auto_expand_indices(list(chs), window_size, per_window, in_channels)
        return proj_dict

    # fallback
    return {}


def _pick_feature_map(features: "OrderedDict[str, torch.Tensor]", pref: str) -> Tuple[str, torch.Tensor]:
    if pref in features:
        return pref, features[pref]
    # fallback: 가장 큰 해상도
    k = max(features.keys(), key=lambda kk: features[kk].shape[-2] * features[kk].shape[-1])
    return k, features[k]


def _compute_gate_gain_from_fmap(
    fmap: torch.Tensor,
    gate_channels: List[int],
    gate_reduce: str,
    region_size: Tuple[int, int],
    score_stride: int,
    gate_gain: float
) -> Optional[torch.Tensor]:
    """
    FPN feature에서 gate_channels만 추출해 채널 축으로 reduce('max'|'mean') 후
    ones-kernel conv로 region mean을 계산하여 (N, oh, ow) 게이트 게인을 반환.
    반환값 g에 대해 최종 score_map *= (1 + gate_gain * g)
    """
    if not gate_channels:
        return None
    N, C, H, W = fmap.shape
    # 채널 서브셋
    sub = fmap[:, gate_channels, :, :]
    # 채널 reduce
    if gate_reduce == "max":
        sub = sub.max(dim=1, keepdim=True).values
    else:
        sub = sub.mean(dim=1, keepdim=True)
    # 윈도우 평균 (conv ones)
    kh, kw = region_size
    ones = torch.ones((1, 1, kh, kw), device=fmap.device, dtype=fmap.dtype)
    g = F.conv2d(sub, ones, stride=score_stride) / float(kh * kw)  # (N,1,oh,ow)
    return (1.0 + gate_gain * g.squeeze(1))  # (N,oh,ow)


# ============================== KBRS Model ==============================

class KBRS_MaskRCNN(MaskRCNN):
    def __init__(self, backbone, num_classes,
                 kbrs_params=None, loss_weights=None):
        super().__init__(backbone, num_classes)
        self.kbrs_params = kbrs_params or {}
        self.loss_weights = loss_weights or {}
        self.static_loss_weights = dict(self.loss_weights)
        self.learnable_loss_weights = self.kbrs_params.get("learnable", None)

        # runtime logging
        self._loss_weight_head = None
        self.log_into_losses = bool(self.kbrs_params.get("log_into_losses", False))
        self.kbrs_last_logs = {}

        # scorer impl
        impl = self.kbrs_params.get('scorer_impl', 'conv')  # 'conv' 권장
        self._scorer_impl = impl

        # scorer args (일부는 forward에서 완성)
        self._scorer_region_size = tuple(self.kbrs_params.get("region_size", (20, 12)))
        self._scorer_stride      = int(self.kbrs_params.get("score_stride", 1))
        self._mixture_tau        = float(self.kbrs_params.get("mixture_tau", 2.0))

        # 게이트 파라미터 (builder에서 계산)
        self._gate_channels_cfg: List[int] = list(self.kbrs_params.get("gate_channels", []))
        self._gate_reduce: str  = str(self.kbrs_params.get("gate_reduce", "max")).lower()
        self._gate_gain: float  = float(self.kbrs_params.get("gate_gain", self.kbrs_params.get("mask_gain", 1.0)))
        self._detach_scorer_input: bool = bool(self.kbrs_params.get("detach_scorer_input", True))

        # projections는 dict로 통일해서 scorer에 전달 (나중에 forward에서 완성)
        self._projections_cfg = self.kbrs_params.get("projections", None)
        self._weights = dict(self.kbrs_params.get("weights", {"density": 1.0, "mixture": 1.0, "centeredness": 1.0}))
        self._mixture_between = self.kbrs_params.get("mixture_between", None)
        self._downsample_before = self.kbrs_params.get("downsample_before", None)

        # scorer placeholder (실제 생성은 forward에서, window-size에 맞춘 projections 정규화 이후)
        self.kbrs_scorer = None

    def _maybe_init_loss_head(self, loss_keys, device):
        if self.learnable_loss_weights and self._loss_weight_head is None:
            self._loss_weight_head = nn.ParameterDict({
                k: nn.Parameter(torch.tensor(1.0, device=device))
                for k in loss_keys
            })

    def forward(self, images, targets=None):
        """
        Full forward (train/eval):
          1) transform
          2) backbone(FPN)
          3) pick FPN feature & compute KBRS score_map
          4) (optional) gate with vision-like channels
          5) RPN -> proposals
          6) ROI heads -> detections
          7) postprocess or assemble losses (+ KBRS losses)
          8) (optional) apply (learnable) loss weights
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # 1) transform to ImageList
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)

        # 2) backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # 3) pick feature map for KBRS
        fmap_key_pref = self.kbrs_params.get("feature_map_name", "smallest")
        fmap_key, fmap = _pick_feature_map(features, fmap_key_pref)

        # KBRS scorer 생성 (window-size에 맞춘 projections/gate 채널 자동 확장 포함)
        # in_channels 추론: 입력 모델의 conv1가 바뀌었으므로 전체 in_channels는 알지만,
        # 여기선 fmap 채널이 필요 없음. 다만 gate/projections 확장은 원본 입력 차원 기준이므로
        # builder에서 넘긴 window_size, per_window(=9 추정)를 활용.
        # 안전하게 per_window는 kbrs_params.get('per_window', 9)로 받음.
        per_window = int(self.kbrs_params.get("per_window", 9))
        window_size = int(self.kbrs_params.get("window_size", 1))
        in_channels_total = per_window * window_size  # 원본 입력 차원 가정

        # projections 정규화
        proj_norm = _normalize_projections(self._projections_cfg, window_size, per_window, in_channels_total)

        # scorer가 아직 없으면 생성
        if self.kbrs_scorer is None:
            if self._scorer_impl == 'conv':
                self.kbrs_scorer = KBRSConvScorer(
                    region_size=self._scorer_region_size,
                    weights=self._weights,
                    projections=proj_norm,
                    mixture_tau=self._mixture_tau,
                    # 내부 mask_channel/mask_gain은 사용하지 않음 (게이트는 builder에서 별도 적용)
                    mask_channel=None,
                    mask_gain=1.0,
                    score_stride=self._scorer_stride,
                    downsample_before=self._downsample_before,
                )
            else:
                # unfold 버전은 메모리 사용량이 크므로 필요할 때만
                self.kbrs_scorer = KBRSUnfoldScorer(
                    region_size=self._scorer_region_size,
                    score_weights=self._weights,
                    projections=proj_norm,
                    mixture_between=self._mixture_between,
                    mask_channel=None
                )

        self.kbrs_scorer = self.kbrs_scorer.to(fmap.device)
        # scorer 입력 detach 옵션
        fmap_for_kbrs = fmap.detach() if self._detach_scorer_input else fmap

        # KBRS score map
        score_map, comp_maps = self.kbrs_scorer(fmap_for_kbrs)

        # 4) (선택) Vision-like 게이트 적용: gate_channels가 있으면 여기서 직접 곱한다
        # gate_channels 자동 확장
        gate_channels = _auto_expand_indices(self._gate_channels_cfg, window_size, per_window, in_channels_total)
        if len(gate_channels) > 0:
            gain = _compute_gate_gain_from_fmap(
                fmap_for_kbrs, gate_channels, self._gate_reduce,
                self._scorer_region_size, self._scorer_stride, self._gate_gain
            )  # (N, oh, ow)
            score_map = score_map * gain
            # 로깅용 컴포넌트에 포함
            comp_maps["gate_gain"] = gain

        # === per-component logging ===
        want_keys = ["density", "mixture", "centeredness", "proj_A", "proj_B", "proj_mixture", "mask", "gate_gain"]
        logs = {}
        for k in want_keys:
            if k in comp_maps:
                stats = _reduce_map_stats(comp_maps[k].detach())
                for stat_name, vec in stats.items():
                    logs[f"{k}/{stat_name}"] = vec
        scalar_logs = {f"{name}_meanB": val.mean() for name, val in logs.items()}
        self.kbrs_last_logs = scalar_logs
        log_losses = {f"log_{k}": v for k, v in scalar_logs.items()} if self.log_into_losses else {}

        # 5) RPN
        proposals, proposal_losses = self.rpn(images, features, targets)

        # 6) ROI heads
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # 7) postprocess (eval)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        if not self.training:
            return detections

        # assemble losses
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # KBRS auxiliary losses
        tau = float(self.kbrs_params.get("loss_scale", 2.0))
        losses["loss_kbrs"] = aux_boost_loss(score_map, tau=tau)
        if bool(self.kbrs_params.get("use_entropy", False)):
            losses["loss_kbrs_entropy"] = aux_entropy_sharp(score_map)

        # (optional) add logs into losses (default weight 0.0)
        if self.log_into_losses:
            losses.update(log_losses)

        # apply weights
        self._maybe_init_loss_head(losses.keys(), images.tensors.device)
        if self.learnable_loss_weights and self._loss_weight_head is not None:
            w = {k: self._loss_weight_head[k] for k in losses.keys()}
            for k in list(losses.keys()):
                losses[k] = losses[k] * w.get(k, torch.tensor(1.0, device=images.tensors.device))
        else:
            for k in list(losses.keys()):
                default_w = 0.0 if k.startswith("log_") else 1.0
                losses[k] = losses[k] * float(self.static_loss_weights.get(k, default_w))

        return losses


# ============================== Builder ==============================

def get_model_instance_segmentation(num_classes: int,
                                    window_size: int = 1,
                                    in_channels: int = None,
                                    do_normalize=False,
                                    use_kbrs=False,
                                    kbrs_params=None,
                                    loss_weights=None):
    """
    window_size > 1 일 때도 자동 호환:
      - transform mean/std 길이 = in_channels
      - gate_channels / projections 의 채널 인덱스가 0~8 범위면 window_size에 맞게 자동 확장 (forward에서 처리)
      - kbrs_params에 window_size, per_window(=9) 주입
    """
    if in_channels is None:
        in_channels = 9 * window_size
    print(f"Using {in_channels} input channels (window size: {window_size})")

    if use_kbrs:
        if kbrs_params is None:
            kbrs_params = {
                "feature_map_name": "smallest",     # 더 작은 FPN맵 사용 권장(OOM 완화)
                "scorer_impl": "conv",              # Conv2d 기반
                "detach_scorer_input": True,        # 백본 그래디언트 차단
                'weights': {"density": 1.0, "mixture": 1.0, "centeredness": 1.0},
                'region_size': (20, 12),
                'learnable': 'static',
                'loss_scale': 2.0,
                'use_entropy': False,
                'projections': [{'name': 'A', 'channels': [0, 1, 2, 3]},
                                {'name': 'B', 'channels': [4, 5, 6, 7]}],
                'mixture_between': ('A', 'B'),
                # 게이트(vision-like): window_size>1이면 자동 확장됨
                'gate_channels': [8],           # 단일 창 인덱스로 주면 자동 확장
                'gate_reduce': 'max',
                'gate_gain': 1.0,
                # scorer 해상도/속도
                "score_stride": 1,
                "downsample_before": None,
                "log_into_losses": False,
                # 아래 두 값은 builder가 주입(자동 확장에 사용)
                # "window_size": window_size,
                # "per_window": 9,
            }

        # 자동 확장에 사용할 메타 정보 주입
        kbrs_params = dict(kbrs_params)
        kbrs_params["window_size"] = int(window_size)
        kbrs_params.setdefault("per_window", 9)

        backbone = resnet_fpn_backbone("resnet50", weights="DEFAULT")
        backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        model = KBRS_MaskRCNN(backbone, num_classes,
                              kbrs_params=kbrs_params,
                              loss_weights=loss_weights)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        model.backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    # === Transform: 채널 수에 맞게 설정 ===
    # 데이터 통계가 없다면 일단 normalize를 끄고 mean=0,std=1로 세팅
    # CustomRCNNTransform(do_normalize=False)이면 내부 normalize를 생략.
    if do_normalize:
        # 사용자가 따로 mean/std를 CustomRCNNTransform 내부에서 제공한다면 그 값을 따름.
        # 여기서는 안전하게 길이만 맞추는 기본값을 둠(나중에 실제 통계로 교체 권장)
        image_mean = [0.0] * in_channels
        image_std  = [1.0] * in_channels
    else:
        image_mean = [0.0] * in_channels
        image_std  = [1.0] * in_channels

    model.transform = CustomRCNNTransform(
        min_size=[800],
        max_size=1333,
        image_mean=image_mean,
        image_std=image_std,
        do_normalize=do_normalize
    )

    # 안전장치
    assert len(model.transform.image_mean) == in_channels, \
        f"image_mean length ({len(model.transform.image_mean)}) != in_channels ({in_channels})"
    assert len(model.transform.image_std) == in_channels, \
        f"image_std length ({len(model.transform.image_std)}) != in_channels ({in_channels})"

    return model
