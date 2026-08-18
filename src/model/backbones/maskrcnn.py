# src/model/backbones/maskrcnn.py
import torch
import torch.nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN

from ..utils.transforms import CustomRCNNTransform

def _kaiming_init_conv(conv: nn.Conv2d):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)

def build_maskrcnn_backbone(
    num_classes: int,
    window_size: int = 1,
    in_channels: int = None,
    do_normalize: bool = False,
    normalize_mean=None,
    normalize_std=None,
    resize_mode: str = "resize",
    min_sizes=None,
    max_size: int = 1333,
    rpn_small_anchors: bool = False,
):
    if in_channels is None:
        in_channels = 9 * window_size

    backbone = resnet_fpn_backbone("resnet50", weights="DEFAULT")
    backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    _kaiming_init_conv(backbone.body.conv1)

    model = MaskRCNN(backbone, num_classes)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced, num_classes)

    if do_normalize and (normalize_mean is not None) and (normalize_std is not None):
        mean = list(normalize_mean)
        std = list(normalize_std)
    else:
        mean, std = None, None

    if resize_mode == "fixed_square":
        min_sizes = (640,)
        max_size = 640
    elif min_sizes is None:
        min_sizes = (800,)

    custom_transform = CustomRCNNTransform(
        in_channels=in_channels,
        min_size=min_sizes,
        max_size=max_size,
        image_mean=mean,
        image_std=std,
    )
    model.transform = custom_transform

    return model
