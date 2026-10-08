import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN
from torchvision.models.detection.roi_heads import RoIHeads

from ..utils.transforms import CustomRCNNTransform

def _kaiming_init_conv(conv: nn.Conv2d):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class SupportTargetRoIHeads(RoIHeads):
    """
    RoIHeads whose box classifier is trained toward a graded target: a
    proposal matched to a ground-truth box with target["mode_weight"] = w gets
    the class distribution (1 - w, w) instead of the one-hot "viewport". With
    mode boxes (mode_targets=soft) w is the mode's support / U, so the box
    score is trained to read as the share of observers on that mode, as the
    heatmap models' peak amplitude is. Background proposals keep (1, 0).

    Only loss_classifier changes. Proposal sampling, box regression, the mask
    branch and inference are torchvision's, so a checkpoint loads into a plain
    MaskRCNN unchanged. The classifier logits are taken from a hook on
    box_predictor rather than by copying RoIHeads.forward; the one-hot loss
    torchvision computes alongside is discarded.
    """

    @classmethod
    def convert(cls, heads: RoIHeads) -> "SupportTargetRoIHeads":
        if heads.box_predictor.cls_score.out_features != 2:
            raise ValueError("SupportTargetRoIHeads assumes background + one class")
        heads.__class__ = cls
        heads._fg_targets = None
        heads._class_logits = None
        heads.box_predictor.register_forward_hook(heads._capture_logits)
        return heads

    def _capture_logits(self, module, inputs, output):
        self._class_logits = output[0]

    def select_training_samples(self, proposals, targets):
        proposals, matched_idxs, labels, regression_targets = super().select_training_samples(proposals, targets)
        fg = []
        for t, idx, lab in zip(targets, matched_idxs, labels):
            if "mode_weight" not in t:
                raise KeyError("SupportTargetRoIHeads needs target['mode_weight'] (train with mode_targets=soft)")
            w = t["mode_weight"].to(device=lab.device, dtype=torch.float32)
            # With no ground truth in the image every sampled proposal is
            # background, and matched_idxs points at a placeholder box.
            w_matched = w[idx] if w.numel() else torch.zeros(lab.shape, device=lab.device)
            fg.append(torch.where(lab > 0, w_matched, torch.zeros_like(w_matched)))
        self._fg_targets = torch.cat(fg)
        return proposals, matched_idxs, labels, regression_targets

    def forward(self, features, proposals, image_shapes, targets=None):
        result, losses = super().forward(features, proposals, image_shapes, targets)
        logits, q = self._class_logits, self._fg_targets
        self._class_logits = self._fg_targets = None
        if self.training:
            soft = torch.stack([1.0 - q, q], dim=1)
            losses["loss_classifier"] = F.cross_entropy(logits, soft)
        return result, losses

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
    soft_mode_cls: bool = False,
):
    if in_channels is None:
        in_channels = 9 * window_size

    backbone = resnet_fpn_backbone(backbone_name="resnet50", weights="DEFAULT")
    backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    _kaiming_init_conv(backbone.body.conv1)

    model = MaskRCNN(backbone, num_classes)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced, num_classes)
    if soft_mode_cls:
        SupportTargetRoIHeads.convert(model.roi_heads)

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
