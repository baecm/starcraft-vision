# src/model/utils/box_ops.py
import torch

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) boxes to (x1, y1, x2, y2) format."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """Convert (x1, y1, x2, y2) boxes to (cx, cy, w, h) format."""
    x1, y1, x2, y2 = x.unbind(-1)
    b = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, (x2 - x1), (y2 - y1)]
    return torch.stack(b, dim=-1)

def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU between two sets of boxes [N, 4] and [M, 4] in (x1, y1, x2, y2) format.
    Returns matrix of shape [N, M].
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)

    lt_c = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, :, 0] * wh_c[:, :, 1]

    giou = iou - (area_c - union) / (area_c + 1e-6)
    return giou
