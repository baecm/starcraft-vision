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
    """Calculates the mean of all values in the patch for stability."""
    return patch.mean()

def score_mixture_torch(patch: torch.Tensor) -> torch.Tensor:
    """Calculates the number of active channels using a differentiable sigmoid approximation."""
    # Use sigmoid as a differentiable approximation for (x > 0)
    channel_active = torch.sigmoid(patch.sum(dim=(1, 2)))
    return channel_active.sum()

def score_centeredness_torch(patch: torch.Tensor) -> torch.Tensor:
    """Calculates the centeredness of the activation in the patch."""
    c, h, w = patch.shape
    y, x = torch.meshgrid(torch.arange(h, device=patch.device), torch.arange(w, device=patch.device), indexing='ij')
    cy, cx = h // 2, w // 2
    sigma = h / 4  # Standard deviation for the Gaussian weight
    weight = torch.exp(-((x - cx).pow(2) + (y - cy).pow(2)) / (2 * sigma ** 2))
    # Using mean instead of sum for stability
    return (patch * weight.unsqueeze(0)).mean()

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
    MaskRCNN model with an additional KBRS loss and flexible loss weighting.
    """
    def __init__(self, backbone, num_classes, kbrs_params, loss_weights=None, **kwargs):
        super().__init__(backbone, num_classes, **kwargs)
        self.kbrs_params = kbrs_params
        self.loss_weights = loss_weights if loss_weights is not None else {}
        
        self.score_funcs = {
            "density": score_density_torch,
            "mixture": score_mixture_torch,
            "centeredness": score_centeredness_torch,
        }
        self.score_fn = build_composite_score_fn_torch(self.score_funcs, self.kbrs_params['weights'])
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

            # Apply all loss weights centrally
            for name, value in losses.items():
                weight = self.loss_weights.get(name, 1.0) # Default weight is 1.0
                losses[name] = value * weight
            
            return losses
        
        return detections

    def compute_kbrs_loss(self, features, targets, image_sizes):
        print(f"DEBUG: compute_kbrs_loss called. Batch size: {len(targets)}")
        num_boxes_per_image = [t['boxes'].shape[0] for t in targets]
        print(f"DEBUG: Number of boxes per image in batch: {num_boxes_per_image}")
        if all(n == 0 for n in num_boxes_per_image):
            print("DEBUG: All images in this batch have 0 boxes. kbrs_loss will be 0.")

        if self.feature_map_name not in features:
            raise ValueError(f"Feature map '{self.feature_map_name}' not found. Available: {list(features.keys())}")
        
        feature_map = features[self.feature_map_name]
        top_k_ratio = self.kbrs_params.get('top_k_ratio', 1.0)
        
        batch_mean_scores = []

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
            
            box_scores = []
            for box in scaled_boxes:
                x1, y1, x2, y2 = box.to(torch.int)
                x1, y1 = x1.clamp(0, feat_w - 1), y1.clamp(0, feat_h - 1)
                x2, y2 = x2.clamp(x1 + 1, feat_w), y2.clamp(y1 + 1, feat_h)
                
                patch = image_feature_map[:, y1:y2, x1:x2]
                if patch.numel() == 0:
                    box_scores.append(torch.tensor(0.0, device=feature_map.device))
                    continue

                # Normalize patch to [0, 1] to make scores independent of activation magnitude
                p_min, p_max = patch.min(), patch.max()
                normalized_patch = (patch - p_min) / (p_max - p_min + 1e-6)

                patch_resized = nn.functional.adaptive_avg_pool2d(
                    normalized_patch.unsqueeze(0), self.kbrs_params['region_size']
                ).squeeze(0)
                
                # Calculate the composite score for the patch
                score = self.score_fn(patch_resized)
                box_scores.append(score)
            
            if not box_scores:
                continue

            # Select top-k scores from the boxes of the current image
            box_scores_tensor = torch.stack(box_scores)
            num_boxes = len(box_scores_tensor)
            k = max(1, int(num_boxes * top_k_ratio))
            
            top_k_scores, _ = torch.topk(box_scores_tensor, k=k, largest=True)
            batch_mean_scores.append(torch.mean(top_k_scores))

        losses = {}
        if not batch_mean_scores:
            losses['loss_kbrs'] = torch.tensor(0.0, device=feature_map.device)
            return losses

        # Calculate the final loss based on the average of the top scores across the batch
        final_score = torch.mean(torch.stack(batch_mean_scores))

        # Clamp final_score to be non-negative to prevent log(<=0) which results in NaN or -inf.
        # This ensures numerical stability for the loss calculation.
        final_score = final_score.clamp(min=0.0)

        print(f"DEBUG: Final composite score (mean of top-k scores): {final_score.item()}")

        # The loss is designed to be inversely proportional to the score.
        # The epsilon prevents division by zero. Clamping above prevents log(<=0).
        losses['loss_kbrs'] = 1.0 / (torch.log(final_score + 1) + 1e-6)
            
        return losses


def get_model_instance_segmentation(num_classes: int, window_size: int, do_normalize=False, use_kbrs=False, kbrs_params=None, loss_weights=None):
    in_channels = 9 * window_size
    
    if use_kbrs:
        if kbrs_params is None:
            kbrs_params = {
                'weights': {"density": 1.0, "mixture": 0.7, "centeredness": 1.2},
                'region_size': (20, 12),
                'feature_map_name': 'pool'
            }
        
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet50', weights=ResNet50_Weights.DEFAULT)
        backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        model = KBRS_MaskRCNN(backbone, num_classes, kbrs_params=kbrs_params, loss_weights=loss_weights)
        
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
