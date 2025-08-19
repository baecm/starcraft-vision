# src/model/maskrcnn_builder.py
import torch.nn as nn
import torchvision

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .CustomRCNNTransform import CustomRCNNTransform
from .KBRS_MaskRCNN import KBRS_MaskRCNN

def get_model_instance_segmentation(num_classes: int,
                                    window_size: int = 1,
                                    in_channels: int = None,
                                    do_normalize=False,
                                    use_kbrs=False,
                                    kbrs_params=None,
                                    loss_weights=None):
    """
    window_size > 1 호환:
      - transform mean/std 길이 = in_channels
      - projections/gate 채널 인덱스가 0~8 범위면 window_size에 맞게 자동 확장 (forward에서 처리)
      - kbrs_params에 window_size, per_window(=9) 주입
    """
    if in_channels is None:
        in_channels = 9 * window_size
    print(f"Using {in_channels} input channels (window size: {window_size})")

    if use_kbrs:
        # 자동 확장 메타 주입
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

    # Transform: 채널 수에 맞춤 (기본 normalize off)
    if do_normalize:
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

    assert len(model.transform.image_mean) == in_channels
    assert len(model.transform.image_std) == in_channels
    return model
