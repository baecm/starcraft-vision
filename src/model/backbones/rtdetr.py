# src/model/backbones/rtdetr.py
import os
import torch
import torch.nn as nn
from ultralytics import RTDETR as UltralyticsRTDETR

def adapt_input_channels(model, in_channels):
    if in_channels == 3:
        return
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            parent_path = name.rsplit('.', 1)
            parent = model if len(parent_path) == 1 else model.get_submodule(parent_path[0])
            attr_name = parent_path[-1]

            new_conv = nn.Conv2d(
                in_channels=in_channels, out_channels=module.out_channels,
                kernel_size=module.kernel_size, stride=module.stride,
                padding=module.padding, dilation=module.dilation,
                groups=module.groups, bias=(module.bias is not None),
                padding_mode=module.padding_mode
            )
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            if new_conv.bias is not None:
                nn.init.zeros_(new_conv.bias)

            setattr(parent, attr_name, new_conv)
            break

def adapt_num_classes(model, num_classes):
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'RTDETRDecoder':
            if getattr(module, 'nc', None) == num_classes:
                break
            module.nc = num_classes
            if hasattr(module, 'enc_score_head'):
                module.enc_score_head = nn.Linear(module.enc_score_head.in_features, num_classes)
            if hasattr(module, 'dec_score_head'):
                for i in range(len(module.dec_score_head)):
                    module.dec_score_head[i] = nn.Linear(module.dec_score_head[i].in_features, num_classes)
            break

class BaseRTDETR(nn.Module):
    def __init__(self, weights: str = "rtdetrv2-l.pt", num_classes: int = 2, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.rtdetr = UltralyticsRTDETR(weights)
        self.model = self.rtdetr.model

        adapt_input_channels(self.model, in_channels)
        adapt_num_classes(self.model, num_classes)

    def _format_targets(self, targets, batched_images):
        _, _, img_h, img_w = batched_images.shape
        batch_idx_list, cls_list, bboxes_list = [], [], []

        for b_idx, target in enumerate(targets):
            boxes = target["boxes"]
            labels = target["labels"]
            if len(boxes) == 0:
                continue

            cls_ids = (labels - 1).float().unsqueeze(1)
            b_ids = torch.full((len(boxes), 1), b_idx, device=boxes.device, dtype=torch.float32)

            cx = (boxes[:, 0] + boxes[:, 2]) / 2.0 / img_w
            cy = (boxes[:, 1] + boxes[:, 3]) / 2.0 / img_h
            w = (boxes[:, 2] - boxes[:, 0]) / img_w
            h = (boxes[:, 3] - boxes[:, 1]) / img_h
            norm_boxes = torch.stack([cx, cy, w, h], dim=1)

            batch_idx_list.append(b_ids)
            cls_list.append(cls_ids)
            bboxes_list.append(norm_boxes)

        if len(batch_idx_list) > 0:
            batch_idx = torch.cat(batch_idx_list, dim=0)
            cls = torch.cat(cls_list, dim=0)
            bboxes = torch.cat(bboxes_list, dim=0)
        else:
            batch_idx = torch.empty((0, 1), device=batched_images.device)
            cls = torch.empty((0, 1), device=batched_images.device)
            bboxes = torch.empty((0, 4), device=batched_images.device)

        return {"batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}

    def forward(self, images, targets=None):
        batched_images = torch.stack(images, dim=0) if isinstance(images, list) else images

        if self.training:
            if targets is None:
                raise ValueError("Targets required in training mode")
            batch_dict = self._format_targets(targets, batched_images)
            preds = self.model._predict_once(batched_images)
            preds_for_loss = preds[:2] if isinstance(preds, tuple) and len(preds) > 2 else preds
            loss_tuple = self.model.criterion(preds_for_loss, batch_dict)
            return {"loss_rtdetr": loss_tuple[0] if isinstance(loss_tuple, tuple) else loss_tuple}
        else:
            preds = self.model(batched_images)
            results = []

            for i, pred in enumerate(preds):
                if hasattr(pred, 'boxes') and pred.boxes is not None:
                    boxes = pred.boxes.xyxy
                    scores = pred.boxes.conf
                    labels = pred.boxes.cls + 1
                else:
                    boxes = torch.empty((0, 4), device=batched_images.device)
                    scores = torch.empty((0,), device=batched_images.device)
                    labels = torch.empty((0,), device=batched_images.device)

                results.append({"boxes": boxes, "scores": scores, "labels": labels})

            return results

def build_rtdetr_backbone(num_classes: int, version: str = "v2", model_size: str = "l", in_channels: int = 3):
    base_cache_dir = os.environ.get("TORCH_CACHE_DIR", "/workspace/.torch_cache")
    weight_dir = os.path.join(base_cache_dir, "ultralytics")
    os.makedirs(weight_dir, exist_ok=True)

    weights_filename = f"rtdetr-{model_size}.pt"
    weights = os.path.join(weight_dir, weights_filename)

    if not os.path.exists(weights):
        import urllib.request
        url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{weights_filename}"
        urllib.request.urlretrieve(url, weights)

    return BaseRTDETR(weights=weights, num_classes=num_classes, in_channels=in_channels)
