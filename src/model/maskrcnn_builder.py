import torch
import torch.nn as nn
import torchvision

from collections import OrderedDict
from typing import Dict, Callable

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models import ResNet50_Weights

from .CustomRCNNTransform import CustomRCNNTransform

# KBRS score functions in torch for differentiability
def score_density_torch(patch: torch.Tensor) -> torch.Tensor:
    """Calculates the sum of all values in the patch."""
    return patch.sum()

def score_mixture_torch(patch: torch.Tensor) -> torch.Tensor:
    """Calculates the number of active channels using a differentiable sigmoid approximation."""
    # Use sigmoid as a differentiable approximation for (x > 0)
    # A large factor is used to make the sigmoid function act like a step function
    channel_active = torch.sigmoid(patch.sum(dim=(1, 2)) * 1e3)
    return channel_active.sum()

def score_centeredness_torch(patch: torch.Tensor) -> torch.Tensor:
    """Calculates the centeredness of the activation in the patch."""
    c, h, w = patch.shape
    y, x = torch.meshgrid(torch.arange(h, device=patch.device), torch.arange(w, device=patch.device), indexing='ij')
    cy, cx = h // 2, w // 2
    sigma = h / 4  # Standard deviation for the Gaussian weight
    weight = torch.exp(-((x - cx).pow(2) + (y - cy).pow(2)) / (2 * sigma ** 2))
    return (patch * weight.unsqueeze(0)).sum()

def build_composite_score_fn_torch(
    score_funcs: Dict[str, Callable[[torch.Tensor], torch.Tensor]],
    weights: Dict[str, float]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Builds a composite score function from a dictionary of score functions and weights."""
    def fn(patch: torch.Tensor) -> torch.Tensor:
        total = torch.tensor(0.0, device=patch.device)
        for name, func in score_funcs.items():
            weight = weights.get(name, 0.0)
            total += weight * func(patch)
        return total
    return fn


class KBRS_MaskRCNN(MaskRCNN):
    """
    MaskRCNN model with an additional KBRS loss.
    The KBRS loss is designed to maximize a score within the ground truth bounding boxes,
    guiding the model to learn more representative features.
    """
    def __init__(self, backbone, num_classes, kbrs_params, **kwargs):
        super().__init__(backbone, num_classes, **kwargs)
        self.kbrs_params = kbrs_params
        
        score_funcs = {
            "density": score_density_torch,
            "mixture": score_mixture_torch,
            "centeredness": score_centeredness_torch,
        }
        self.score_fn = build_composite_score_fn_torch(score_funcs, self.kbrs_params['weights'])
        self.kbrs_loss_weight = self.kbrs_params.get('loss_weight', 1.0)
        self.feature_map_name = self.kbrs_params.get('feature_map_name', 'pool')

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            kbrs_losses = self.compute_kbrs_loss(features, targets, images.image_sizes)
            losses.update(kbrs_losses)
            return losses
        
        return detections

    def compute_kbrs_loss(self, features, targets, image_sizes):
        if self.feature_map_name not in features:
            raise ValueError(f"Feature map '{self.feature_map_name}' not found. Available: {list(features.keys())}")
        
        feature_map = features[self.feature_map_name]
        
        # 각 score 구성 요소별로 점수를 저장할 리스트
        scores_by_type = {name: [] for name in self.score_fn.score_funcs.keys()}
        
        for i, target in enumerate(targets):
            gt_boxes = target['boxes']
            if gt_boxes.shape[0] == 0:
                continue

            img_h, img_w = image_sizes[i]
            feat_h, feat_w = feature_map.shape[-2:]
            
            scale_w = feat_w / img_w
            scale_h = feat_h / img_h

            scaled_boxes = gt_boxes.clone()
            scaled_boxes[:, 0::2] *= scale_w
            scaled_boxes[:, 1::2] *= scale_h
            
            image_feature_map = feature_map[i]

            for box in scaled_boxes:
                x1, y1, x2, y2 = box.to(torch.int)
                x1, y1 = x1.clamp(0, feat_w - 1), y1.clamp(0, feat_h - 1)
                x2, y2 = x2.clamp(x1 + 1, feat_w), y2.clamp(y1 + 1, feat_h)
                
                patch = image_feature_map[:, y1:y2, x1:x2]
                if patch.numel() == 0:
                    continue

                patch_resized = nn.functional.adaptive_avg_pool2d(
                    patch.unsqueeze(0), self.kbrs_params['region_size']
                ).squeeze(0)
                
                # 각 score 함수를 개별적으로 호출하여 점수 계산
                for name, func in self.score_fn.score_funcs.items():
                    scores_by_type[name].append(func(patch_resized))
        
        # 최종 loss를 담을 딕셔너리
        losses = {}
        total_score = torch.tensor(0.0, device=feature_map.device)

        for name, scores in scores_by_type.items():
            if not scores:
                # 해당 타입의 score가 없는 경우 loss를 0으로 설정
                avg_score = torch.tensor(0.0, device=feature_map.device)
            else:
                avg_score = torch.mean(torch.stack(scores))

            # 개별 loss 계산 (score가 높을수록 loss가 낮아지도록)
            # 1e-6은 분모가 0이 되는 것을 방지
            individual_loss = 1.0 / (avg_score + 1e-6)
            losses[f'loss_kbrs_{name}'] = individual_loss
            
            # 가중치를 적용하여 전체 score에 합산
            total_score += self.score_fn.weights.get(name, 0.0) * avg_score

        # 전체 KBRS loss 계산
        if not any(scores_by_type.values()):
            losses['loss_kbrs'] = torch.tensor(0.0, device=feature_map.device)
        else:
            losses['loss_kbrs'] = (1.0 / (total_score + 1e-6)) * self.kbrs_loss_weight
            
        return losses


def get_model_instance_segmentation(num_classes: int, window_size: int, do_normalize=False, use_kbrs=False, kbrs_params=None):
    in_channels = 9 * window_size
    
    if use_kbrs:
        if kbrs_params is None:
            kbrs_params = {
                'weights': {"density": 1.0, "mixture": 0.7, "centeredness": 1.2},
                'loss_weight': 0.5,
                'region_size': (20, 12),
                'feature_map_name': 'pool'
            }
        
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet50', weights=ResNet50_Weights.DEFAULT)
        backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        model = KBRS_MaskRCNN(backbone, num_classes, kbrs_params=kbrs_params)
        
        # Replace the pre-trained heads with new ones for the given num_classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    else:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
        
        model.backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    # Common transform for both model types
    model.transform = CustomRCNNTransform(
        min_size=[800],
        max_size=1333,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        do_normalize=do_normalize
    )

    return model

if __name__ == "__main__":
    # Example of creating a standard model
    print("--- Standard MaskRCNN ---")
    model_std = get_model_instance_segmentation(num_classes=2, window_size=1, use_kbrs=False)
    print(model_std)

    # Example of creating a model with KBRS loss
    print("--- MaskRCNN with KBRS Loss ---")
    model_kbrs = get_model_instance_segmentation(num_classes=2, window_size=1, use_kbrs=True)
    print(model_kbrs)
    
    model_kbrs.train()
    
    images = [torch.rand(9, 800, 800)]
    targets = [{
        "boxes": torch.rand(2, 4) * 400,
        "labels": torch.randint(0, 2, (2,)),
        "masks": torch.rand(2, 800, 800)
    }]
    
    # The model in training mode returns a dictionary of losses
    loss_dict = model_kbrs(images, targets)
    print("KBRS Model Loss Dictionary:")
    print(loss_dict)
