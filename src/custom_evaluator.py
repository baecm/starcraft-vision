from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pycocotools.mask as mask_util
from pycocotools.coco import COCO


# ----------------------------------------------------------------------
# Dataclass: per-image 결과 구조
# ----------------------------------------------------------------------


@dataclass
class ImageIR:
    image_id: int
    width: int
    height: int
    ir: float  # intersection ratio
    overlap_count: int  # GT와 겹치는 예측 개수
    density: float
    centeredness: float
    mixture: float


# ----------------------------------------------------------------------
# 내부 유틸리티: RLE / bbox / window 계산
# ----------------------------------------------------------------------


def _poly_to_rle(poly, h: int, w: int) -> dict:
    """
    segmentation이 polygon(list[list])인 경우 RLE로 변환.
    """
    if isinstance(poly, dict) and "counts" in poly:
        rle = poly
        if isinstance(rle["counts"], list):
            rle = mask_util.frPyObjects(rle, h, w)
        return rle
    if isinstance(poly, list):
        rles = mask_util.frPyObjects(poly, h, w)
        rle = mask_util.merge(rles)
        return rle
    raise ValueError(f"Unsupported segmentation format: {type(poly)}")


def _anno_to_rle(anno: dict, h: int, w: int) -> dict:
    """
    annotation dict → RLE mask
    - segmentation 있으면 그걸 사용
    - 없으면 bbox로 사각형 mask 생성
    """
    if "segmentation" in anno and anno["segmentation"] is not None:
        seg = anno["segmentation"]
        if isinstance(seg, list):
            rle = _poly_to_rle(seg, h, w)
        elif isinstance(seg, dict):
            rle = seg
            if isinstance(seg.get("counts", None), list):
                rle = mask_util.frPyObjects(rle, h, w)
        else:
            raise ValueError("Unknown segmentation type in annotation.")
        return rle

    if "bbox" in anno and anno["bbox"] is not None:
        x, y, wbox, hbox = anno["bbox"]
        x0 = max(0, int(np.floor(x)))
        y0 = max(0, int(np.floor(y)))
        x1 = min(w, int(np.ceil(x + wbox)))
        y1 = min(h, int(np.ceil(y + hbox)))
        if x1 <= x0 or y1 <= y0:
            return mask_util.encode(
                np.zeros((h, w), dtype=np.uint8, order="F")
            )
        m = np.zeros((h, w), dtype=np.uint8)
        m[y0:y1, x0:x1] = 1
        rle = mask_util.encode(np.asfortranarray(m))
        return rle

    raise ValueError("Annotation lacks both 'segmentation' and 'bbox'.")


def _union_rles(rles: List[dict], h: int, w: int) -> dict:
    if not rles:
        return mask_util.encode(
            np.zeros((h, w), dtype=np.uint8, order="F")
        )
    return mask_util.merge(rles)


def _area(rle: dict) -> float:
    return float(mask_util.area(rle))


def _intersect(rle1: dict, rle2: dict) -> dict:
    return mask_util.merge([rle1, rle2], intersect=True)


def _bbox_from_rle(rle: dict) -> Tuple[int, int, int, int]:
    """
    RLE → bbox (x0, y0, x1, y1), 정수 좌표
    """
    bb = mask_util.toBbox(rle)  # (x,y,w,h) float
    x, y, w, h = bb
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = int(np.ceil(x + w))
    y1 = int(np.ceil(y + h))
    return x0, y0, x1, y1


def _centroid_from_rle(rle: dict) -> Tuple[float, float]:
    """
    mask의 중심점을 bbox의 중점으로 근사.
    """
    x0, y0, x1, y1 = _bbox_from_rle(rle)
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _window_mask(
    center_x: float,
    center_y: float,
    win_w: int,
    win_h: int,
    H: int,
    W: int,
) -> Tuple[dict, float, Tuple[int, int, int, int]]:
    """
    (center_x, center_y)를 중심으로 하는 win_w x win_h 윈도우를
    이미지 (H,W) 안에 clip해서 binary mask + 영역 출력.
    """
    half_w = win_w // 2
    half_h = win_h // 2
    x0 = int(np.round(center_x)) - half_w
    y0 = int(np.round(center_y)) - half_h
    x1 = x0 + win_w
    y1 = y0 + win_h

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W, x1)
    y1 = min(H, y1)

    if x1 <= x0 or y1 <= y0:
        m = np.zeros((H, W), dtype=np.uint8)
        return mask_util.encode(np.asfortranarray(m)), 0.0, (x0, y0, x1, y1)

    m = np.zeros((H, W), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return mask_util.encode(np.asfortranarray(m)), float((x1 - x0) * (y1 - y0)), (
        x0,
        y0,
        x1,
        y1,
    )


def _window_center(window_source: str, P: Optional[dict], G: dict) -> Tuple[float, float]:
    """
    커널(윈도우)의 중심을 어디에 둘지 결정:
    - window_source == "pred"  & P가 있으면: 예측 중심
    - 그 외: GT 중심
    """
    if window_source == "pred" and P is not None:
        return _centroid_from_rle(P)
    return _centroid_from_rle(G)


# ----------------------------------------------------------------------
# Intersection ratio 및 kernel 기반 score
# ----------------------------------------------------------------------


def intersection_ratio(P: dict, G: dict, denom: str) -> float:
    """
    P(예측 mask)와 G(GT mask)의 intersection ratio.
    denom:
      - "gt"    : |P∩G| / |G|
      - "pred"  : |P∩G| / |P|
      - "union" : |P∩G| / |P∪G|
    """
    inter = _intersect(P, G)
    inter_area = _area(inter)
    if denom == "gt":
        D = _area(G)
    elif denom == "pred":
        D = _area(P)
    elif denom == "union":
        D = _area(P) + _area(G) - inter_area
    else:
        raise ValueError("Unknown denom, choose from {'gt','pred','union'}")
    if D <= 0:
        return 0.0
    return float(inter_area / D)


def kernel_scores(
    center_x: float,
    center_y: float,
    win_w: int,
    win_h: int,
    H: int,
    W: int,
    T_mask: dict,
) -> Tuple[float, float, float]:
    """
    윈도우 내부에서:
      - density      : 윈도우 면적 대비 mask 비율
      - centeredness : 윈도우 중심과 mask 중심 거리 기반 (0~1)
      - mixture      : density * centeredness
    """
    Wmask, Warea, (x0, y0, x1, y1) = _window_mask(center_x, center_y, win_w, win_h, H, W)
    if Warea <= 0:
        return 0.0, 0.0, 0.0

    TinW = _intersect(T_mask, Wmask)
    t_area = _area(TinW)
    density = float(t_area / Warea)

    # centeredness: T∩W의 centroid와 윈도우 중심 간 거리
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    if t_area <= 0:
        centeredness = 0.0
    else:
        tcx, tcy = _centroid_from_rle(TinW)
        dist = float(np.hypot(tcx - cx, tcy - cy))
        half_diag = float(np.hypot((x1 - x0), (y1 - y0)) / 2.0)
        centeredness = max(0.0, 1.0 - (dist / half_diag)) if half_diag > 0 else 0.0

    mixture = density * centeredness
    return density, centeredness, mixture


# ----------------------------------------------------------------------
# 메인: eval_intersection_run
# ----------------------------------------------------------------------


def eval_intersection_run(
    coco_gt: COCO,
    preds: List[dict],
    denom: str = "gt",
    pred_agg: str = "best",
    cat_id: Optional[int] = None,
    window_source: str = "pred",
    window_target: str = "gt",
    win_w: int = 20,
    win_h: int = 12,
) -> Tuple[List[ImageIR], Dict[str, float]]:
    """
    Observer-style Intersection / kernel metric 계산의 코어 함수.

    Args:
        coco_gt: pycocotools COCO (GT).
        preds  : COCO detection 리스트
                 (각 원소는 {image_id, category_id, score, bbox/segmentation} dict)
        denom  : IR 분모 ("gt" / "pred" / "union")
        pred_agg:
            - "best"  : 이미지별로 IR이 가장 큰 하나의 예측만 사용
            - "union" : 모든 예측을 union한 mask로 IR 계산
        cat_id : 단일 category만 보고 싶을 때 필터 (None이면 전체)
        window_source: 커널 중심 위치 ("pred" or "gt")
        window_target: 커널 내부에서 사용할 mask ("gt" or "intersect")
        win_w, win_h : 커널 크기 (픽셀 단위)
    Returns:
        per_image: ImageIR 리스트 (이미지 단위 결과)
        aggregates: 전체 데이터셋에 대한 집계 값 dict
    """
    # GT를 image_id별로 묶기
    gt_by_img: Dict[int, List[dict]] = {}
    for ann in coco_gt.dataset["annotations"]:
        if cat_id is not None and ann.get("category_id") != cat_id:
            continue
        gt_by_img.setdefault(ann["image_id"], []).append(ann)

    # 이미지 메타 (H,W)
    img_meta: Dict[int, Tuple[int, int]] = {}
    for img in coco_gt.dataset["images"]:
        img_meta[img["id"]] = (img["height"], img["width"])

    # pred를 image_id별로 묶기
    pred_by_img: Dict[int, List[dict]] = {}
    for d in preds:
        if cat_id is not None and d.get("category_id") != cat_id:
            continue
        pred_by_img.setdefault(d["image_id"], []).append(d)

    per_image: List[ImageIR] = []

    for image_id, (H, W) in img_meta.items():
        gt_list = gt_by_img.get(image_id, [])
        if len(gt_list) == 0:
            continue

        # 모든 GT를 union한 mask
        G = _union_rles([_anno_to_rle(ann, H, W) for ann in gt_list], H, W)
        preds_list = pred_by_img.get(image_id, [])

        oc = 0           # overlap_count
        ir_val = 0.0     # intersection ratio
        P_sel = None     # 윈도우 기준에 사용할 P

        if len(preds_list) > 0:
            # GT와 실제로 겹치는 예측 개수
            for p in preds_list:
                Pp = _anno_to_rle(p, H, W)
                if _area(_intersect(Pp, G)) > 0:
                    oc += 1

            # 이미지별 예측 집계 방식
            if pred_agg == "union":
                P_union = _union_rles(
                    [_anno_to_rle(p, H, W) for p in preds_list],
                    H,
                    W,
                )
                ir_val = intersection_ratio(P_union, G, denom)
                P_sel = P_union
            elif pred_agg == "best":
                ir_best = 0.0
                P_best = None
                for p in preds_list:
                    Pp = _anno_to_rle(p, H, W)
                    irp = intersection_ratio(Pp, G, denom)
                    if irp > ir_best:
                        ir_best = irp
                        P_best = Pp
                ir_val = ir_best
                P_sel = P_best
            else:
                raise ValueError("pred_agg must be 'best' or 'union'")

        # 커널 중심/타겟 mask 선택
        cx, cy = _window_center(window_source, P_sel, G)
        if window_target == "gt":
            Tmask = G
        elif window_target == "intersect" and P_sel is not None:
            Tmask = _intersect(P_sel, G)
        else:
            Tmask = G  # fallback

        dens, cent, mix = kernel_scores(cx, cy, win_w, win_h, H, W, Tmask)

        per_image.append(
            ImageIR(
                image_id=image_id,
                width=W,
                height=H,
                ir=ir_val,
                overlap_count=oc,
                density=dens,
                centeredness=cent,
                mixture=mix,
            )
        )

    if len(per_image) == 0:
        raise RuntimeError(
            "No images with GT found. Check category filter or inputs."
        )

    # 집계값 계산
    ir_values = np.array([x.ir for x in per_image], dtype=float)
    mean_ir = float(np.mean(ir_values))
    median_ir = float(np.median(ir_values))
    p90_ir = float(np.percentile(ir_values, 90))
    coverage_any = float(np.mean(ir_values > 0.0))
    multi_cov = float(np.mean(np.array([x.overlap_count for x in per_image]) >= 2))

    dens_vals = np.array([x.density for x in per_image], dtype=float)
    cent_vals = np.array([x.centeredness for x in per_image], dtype=float)
    mix_vals = np.array([x.mixture for x in per_image], dtype=float)

    aggregates: Dict[str, float] = {
        "mean_ir": mean_ir,
        "median_ir": median_ir,
        "p90_ir": p90_ir,
        "coverage_any": coverage_any,
        "multi_coverage": multi_cov,
        "num_images": int(len(per_image)),
        "mean_density": float(np.mean(dens_vals)),
        "mean_centeredness": float(np.mean(cent_vals)),
        "mean_mixture": float(np.mean(mix_vals)),
    }
    return per_image, aggregates


# 기존 인터페이스 호환용 alias
def eval_run(
    coco_gt: COCO,
    preds: List[dict],
    denom: str = "gt",
    pred_agg: str = "best",
    cat_id: Optional[int] = None,
    window_source: str = "pred",
    window_target: str = "gt",
    win_w: int = 20,
    win_h: int = 12,
):
    return eval_intersection_run(
        coco_gt=coco_gt,
        preds=preds,
        denom=denom,
        pred_agg=pred_agg,
        cat_id=cat_id,
        window_source=window_source,
        window_target=window_target,
        win_w=win_w,
        win_h=win_h,
    )
