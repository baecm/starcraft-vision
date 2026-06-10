import torch
import torch.nn as nn
from ultralytics import RTDETR as Ultralytics_RTDETR

# ==========================================
# 유틸리티 함수: 모델 구조 개조
# ==========================================
def adapt_input_channels(model, in_channels):
    if in_channels == 3: return
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
            print(f"[Info] Replaced backbone input layer '{name}' to accept {in_channels} channels.")
            break

def adapt_num_classes(model, num_classes):
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'RTDETRDecoder':
            if getattr(module, 'nc', None) == num_classes: break
            
            print(f"[Info] Adapting classification head from {module.nc} to {num_classes} classes.")
            module.nc = num_classes
            
            if hasattr(module, 'enc_score_head'):
                module.enc_score_head = nn.Linear(module.enc_score_head.in_features, num_classes)
            if hasattr(module, 'dec_score_head'):
                for i in range(len(module.dec_score_head)):
                    module.dec_score_head[i] = nn.Linear(module.dec_score_head[i].in_features, num_classes)
            break

# ==========================================
# 부모 클래스: Base RT-DETR
# ==========================================
class BaseRTDETR(nn.Module):
    def __init__(self, weights="rtdetrv2-l.pt", num_classes=2, in_channels=3):
        super().__init__()
        self.version = "v2" if "v2" in weights.lower() else "v1"
        
        temp_wrapper = Ultralytics_RTDETR(weights)
        self.model = temp_wrapper.model
        
        adapt_input_channels(self.model, in_channels)        
        adapt_num_classes(self.model, num_classes)
        
        for param in self.model.parameters():
            param.requires_grad = True
            
        try:
            from ultralytics.models.utils.loss import RTDETRDetectionLoss
        except ImportError:
            from ultralytics.utils.loss import RTDETRDetectionLoss
            
        self.criterion = RTDETRDetectionLoss(nc=num_classes, use_vfl=True)

    def _format_targets(self, targets, batched_images):
        device = batched_images.device
        img_h, img_w = batched_images.shape[-2:]
        batch_idx_list, cls_list, bboxes_list, gt_groups = [], [], [], []

        for b_idx, target in enumerate(targets):
            boxes = target["boxes"]
            gt_groups.append(len(boxes))

            if len(boxes) == 0: continue

            x1, y1, x2, y2 = boxes.unbind(1)
            boxes_cxcywh = torch.stack(((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1), dim=1)
            boxes_cxcywh[:, [0, 2]] /= img_w
            boxes_cxcywh[:, [1, 3]] /= img_h

            batch_idx_list.append(torch.full((len(boxes),), b_idx, dtype=torch.long, device=device))
            cls_list.append((target["labels"] - 1).to(device))
            bboxes_list.append(boxes_cxcywh.to(device))

        if len(batch_idx_list) > 0:
            batch_idx = torch.cat(batch_idx_list, dim=0)
            cls = torch.cat(cls_list, dim=0).long()
            bboxes = torch.cat(bboxes_list, dim=0).float()
        else:
            batch_idx = torch.empty((0,), dtype=torch.long, device=device)
            cls = torch.empty((0,), dtype=torch.long, device=device)
            bboxes = torch.empty((0, 4), dtype=torch.float, device=device)

        return {
            'img': batched_images, 'batch_idx': batch_idx,
            'cls': cls, 'bboxes': bboxes, 'gt_groups': tuple(gt_groups) 
        }
        
    def _format_predictions(self, preds, image_shape):
        img_h, img_w = image_shape
        
        # 1. Ultralytics의 원시 추론 결과는 보통 튜플이거나 3차원 텐서입니다.
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[0]
            
        # preds shape: (Batch, Num_Queries, 4 + Num_Classes)
        # 만약 구조상 채널이 중간에 있다면 (Batch, 4 + Num_Classes, Num_Queries) 형태를 뒤집어줍니다.
        if preds.shape[1] == 4 + self.criterion.nc:
            preds = preds.transpose(1, 2)
            
        results = []
        # 2. 배치(Batch) 내의 각 이미지 단위로 분리해서 처리
        for p in preds: 
            # p shape: (300, 6) -> 앞의 4개는 박스 좌표, 뒤의 2개는 클래스 점수
            boxes = p[:, :4]
            cls_scores = p[:, 4:]
            
            # 각 박스별로 가장 높은 점수와 그 때의 클래스 라벨(0 또는 1)을 추출
            scores, labels = torch.max(cls_scores, dim=-1)
            
            # 3. 박스 좌표 변환: Normalized [cx, cy, w, h] -> Absolute [x1, y1, x2, y2]
            cx, cy, w, h = boxes.unbind(-1)
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            
            boxes_xyxy = torch.stack((x1, y1, x2, y2), dim=-1)
            
            # 4. 라벨 복원: 학습 파트(_format_targets)에서 1을 뺐으므로, 추론 시에는 다시 1을 더해 복원합니다.
            labels = labels + 1 
            
            # 5. 추론 스크립트가 기대하는 Mask R-CNN 스타일의 딕셔너리로 포장
            results.append({
                "boxes": boxes_xyxy,
                "scores": scores,
                "labels": labels
            })
            
        return results

# ==========================================
# 자식 클래스 1: 바닐라 RT-DETR
# ==========================================
class CustomRTDETR(BaseRTDETR):
    def forward(self, images, targets=None):
        batched_images = torch.stack(images, dim=0) if isinstance(images, list) else images
            
        if self.training:
            if targets is None: raise ValueError("In training mode, targets should be passed")
            
            batch_dict = self._format_targets(targets, batched_images)
            preds = self.model._predict_once(batched_images)
            preds_for_loss = preds[:2] if isinstance(preds, tuple) and len(preds) > 2 else preds
                
            criterion_out = self.criterion(preds_for_loss, batch_dict)
            
            if isinstance(criterion_out, dict): return criterion_out
            elif isinstance(criterion_out, tuple): return {f"loss_rtdetr_{self.version}": criterion_out[0]}
            else: return {f"loss_rtdetr_{self.version}": criterion_out}
            
        else:
            with torch.no_grad():
                preds = self.model(batched_images)
            return self._format_predictions(preds, batched_images.shape[-2:])