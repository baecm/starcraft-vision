# src/model/maskrcnn_builder.py
import torch.nn as nn
import torchvision

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .CustomRCNNTransform import CustomRCNNTransform
from .KBRS_MaskRCNN import KBRS_MaskRCNN


def _kaiming_init_conv(conv: nn.Conv2d):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def get_model_instance_segmentation(num_classes: int,
                                    window_size: int = 1,
                                    in_channels: int = None,
                                    do_normalize: bool = False,
                                    normalize_mean=None,
                                    normalize_std=None,
                                    resize_mode: str = "resize",     # "resize" or "keep"
                                    min_sizes=None,                  # e.g. [800] or [640,800,896,960,1024]
                                    max_size: int = 1333,
                                    rpn_small_anchors: bool = False, # 원본크기 유지 시 권장
                                    use_kbrs: bool = False,
                                    kbrs_params=None,
                                    loss_weights=None):
    """
    - in_channels 미지정 시 9 * window_size
    - 첫 conv를 in_channels로 교체 + Kaiming init
    - resize_mode / min_sizes / max_size로 리사이즈 제어
    - do_normalize=True면 mean/std 사용(길이는 in_channels와 동일해야 함)
    - rpn_small_anchors=True면 작은 입력용 앵커로 교체
    """
    if in_channels is None:
        in_channels = 9 * window_size
    print(f"Using {in_channels} input channels (window size: {window_size})")

    if use_kbrs:
        # 자동 확장 메타 주입
        kbrs_params = dict(kbrs_params or {})
        kbrs_params["window_size"] = int(window_size)
        kbrs_params.setdefault("per_window", 9)

        backbone = resnet_fpn_backbone("resnet50", weights="DEFAULT")
        backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        _kaiming_init_conv(backbone.body.conv1)

        model = KBRS_MaskRCNN(backbone, num_classes,
                              kbrs_params=kbrs_params,
                              loss_weights=loss_weights)
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        model.backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        _kaiming_init_conv(model.backbone.body.conv1)

    # heads
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    # 작은 앵커 (리사이즈 끌 때 유리)
    if rpn_small_anchors:
        from torchvision.models.detection.rpn import AnchorGenerator
        anchor_generator = AnchorGenerator(
            sizes=((8,), (16,), (32,), (64,), (128,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )
        model.rpn.anchor_generator = anchor_generator

    # --- Transform 설정 ---
    C = in_channels
    if do_normalize:
        image_mean = (normalize_mean if normalize_mean is not None else [0.0] * C)
        image_std  = (normalize_std  if normalize_std  is not None else [1.0] * C)
    else:
        image_mean = [0.0] * C
        image_std  = [1.0] * C

    if min_sizes is None:
        min_sizes = [800] if resize_mode == "resize" else [128]

    # resize_mode="keep"이면 resize를 스킵
    do_resize = (resize_mode == "resize")

    model.transform = CustomRCNNTransform(
        min_size=min_sizes,
        max_size=max_size,
        image_mean=image_mean,
        image_std=image_std,
        do_normalize=do_normalize,
        do_resize=do_resize
    )

    assert len(model.transform.image_mean) == in_channels
    assert len(model.transform.image_std) == in_channels
    return model
