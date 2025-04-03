import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.nn as nn

def get_model_instance_segmentation(num_classes: int, in_channels: int = 36):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.backbone.body.conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model