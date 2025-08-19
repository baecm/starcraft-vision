# src/model/kbrs_eval.py
import torch
from typing import List, Tuple, Dict

def _scoremap_topk(score_map: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # score_map: (oh, ow)
    vals, idx = torch.topk(score_map.flatten(), k=min(k, score_map.numel()))
    ys = idx // score_map.shape[1]
    xs = idx %  score_map.shape[1]
    return vals, ys, xs

def nms_boxes(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float=0.5) -> List[int]:
    if boxes.numel() == 0: return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1: break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-8)
        order = rest[iou <= iou_thr]
    return keep

def windows_from_score_map(score_map: torch.Tensor,    # (oh, ow)
                           H: int, W: int,
                           kernel_size: Tuple[int,int],
                           stride: int | Tuple[int,int] = 1,
                           padding: int | Tuple[int,int] = 0,
                           dilation: int | Tuple[int,int] = 1,
                           topk: int = 50,
                           nms_iou: float = 0.5) -> Tuple[List[Tuple[int,int,int,int]], List[float]]:
    """ score_map 위치를 원영상 윈도우 박스로 복원하고 NMS """
    if isinstance(stride, int):   stride   = (stride, stride)
    if isinstance(padding, int):  padding  = (padding, padding)
    if isinstance(dilation, int): dilation = (dilation, dilation)
    kh, kw = kernel_size
    eff_h = dilation[0]*(kh-1)+1
    eff_w = dilation[1]*(kw-1)+1

    vals, ys, xs = _scoremap_topk(score_map, topk)
    boxes = []
    scores = []
    for v, y, x in zip(vals, ys, xs):
        y0 = int(max(0, y.item()*stride[0] - padding[0]))
        x0 = int(max(0, x.item()*stride[1] - padding[1]))
        y1 = int(min(H, y0 + eff_h))
        x1 = int(min(W, x0 + eff_w))
        if y1 > y0 and x1 > x0:
            boxes.append((x0, y0, x1, y1))
            scores.append(float(v.item()))

    # NMS on boxes (IoU of rectangles)
    if len(boxes) == 0:
        return [], []
    keep = nms_boxes(torch.tensor(boxes, dtype=torch.float32), torch.tensor(scores), iou_thr=nms_iou)
    boxes = [boxes[i] for i in keep]
    scores = [scores[i] for i in keep]
    return boxes, scores

def rasterize_windows_to_masks(boxes: List[Tuple[int,int,int,int]], H: int, W: int, device=None) -> List[torch.Tensor]:
    ms = []
    for (x0,y0,x1,y1) in boxes:
        m = torch.zeros((H,W), device=device)
        m[y0:y1, x0:x1] = 1.0
        ms.append(m)
    return ms

def masks_to_tensor(masks: List[torch.Tensor]) -> torch.Tensor:
    if len(masks) == 0: return torch.zeros((0,1,1))
    return torch.stack([m.float() for m in masks], dim=0)  # (N,H,W)

def pairwise_mask_iou(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    # pred: (P,H,W), gt: (G,H,W) -> (P,G)
    if pred.numel() == 0 or gt.numel() == 0:
        return torch.zeros((pred.shape[0], gt.shape[0]), device=pred.device)
    P, H, W = pred.shape
    G = gt.shape[0]
    pf = pred.reshape(P, -1); gf = gt.reshape(G, -1)
    inter = pf @ gf.T
    ua = pf.sum(1, keepdim=True) + gf.sum(1).unsqueeze(0) - inter
    return inter / (ua + 1e-8)

def match_greedy_by_iou(iou: torch.Tensor, thr: float) -> Tuple[List[int], List[int], List[float]]:
    P, G = iou.shape
    used_gt = torch.zeros(G, dtype=torch.bool, device=iou.device)
    mp, mg, mi = [], [], []
    for p in range(P):
        # GT 중 아직 사용되지 않은 것들에 대해 최대 IoU 선택
        mask = (~used_gt).float()
        best_iou, best_g = (iou[p] * mask).max(dim=0)
        if best_iou.item() >= thr and not used_gt[best_g]:
            used_gt[best_g] = True
            mp.append(p); mg.append(best_g.item()); mi.append(best_iou.item())
    return mp, mg, mi

def eval_image_instances(pred_masks: List[torch.Tensor], pred_scores: List[float],
                         gt_masks: List[torch.Tensor], iou_thr: float=0.5) -> Dict[str,float]:
    device = (pred_masks[0].device if pred_masks else (gt_masks[0].device if gt_masks else "cpu"))
    P, G = len(pred_masks), len(gt_masks)
    if P==0 and G==0: return {"AP":1.0,"Precision":1.0,"Recall":1.0,"mIoU_matched":1.0}
    if P==0 or G==0:  return {"AP":0.0,"Precision":0.0,"Recall":0.0,"mIoU_matched":0.0}
    order = torch.tensor(pred_scores, device=device).argsort(descending=True)
    pred = masks_to_tensor([pred_masks[i] for i in order]).to(device)
    gt   = masks_to_tensor(gt_masks).to(device)
    iou = pairwise_mask_iou(pred, gt)
    mp, mg, mi = match_greedy_by_iou(iou, thr=iou_thr)
    TP = len(mp); FP = pred.shape[0] - TP; FN = gt.shape[0] - TP
    prec = TP / (TP + FP + 1e-8); rec = TP / (TP + FN + 1e-8)
    AP = prec  # 단일 스냅숏 근사
    mIoU = float(sum(mi)/max(1,len(mi)))
    return {"AP": float(AP), "Precision": float(prec), "Recall": float(rec), "mIoU_matched": mIoU}

def eval_image_instances_multi_thr(pred_masks: List[torch.Tensor], pred_scores: List[float],
                                   gt_masks: List[torch.Tensor],
                                   iou_thrs = [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]) -> Dict[str,float]:
    res = [eval_image_instances(pred_masks, pred_scores, gt_masks, t) for t in iou_thrs]
    mAP = sum(r["AP"] for r in res)/len(res)
    out = {f"AP@{t:.2f}": res[i]["AP"] for i,t in enumerate(iou_thrs)}
    out["mAP@[.50:.95]"] = float(mAP)
    out["mPrecision"] = float(sum(r["Precision"] for r in res)/len(res))
    out["mRecall"]    = float(sum(r["Recall"] for r in res)/len(res))
    out["meanMatchedIoU"] = float(sum(r["mIoU_matched"] for r in res)/len(res))
    return out
