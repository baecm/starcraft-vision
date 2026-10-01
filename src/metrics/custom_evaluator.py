# src/metrics/custom_evaluator.py
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
    overlap_count: int  # GT와 겹치는 예측 개수 or nonzero pixel count in legacy


# ----------------------------------------------------------------------
# 내부 유틸리티: RLE / bbox / window 계산 (modern path)
# ----------------------------------------------------------------------


def _poly_to_rle(poly, h: int, w: int) -> dict:
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
            return mask_util.encode(np.asfortranarray(np.zeros((h, w), dtype=np.uint8)))
        m = np.zeros((h, w), dtype=np.uint8)
        m[y0:y1, x0:x1] = 1
        rle = mask_util.encode(np.asfortranarray(m))
        return rle

    raise ValueError("Annotation lacks both 'segmentation' and 'bbox'.")


def _union_rles(rles: List[dict], h: int, w: int) -> dict:
    if not rles:
        return mask_util.encode(np.asfortranarray(np.zeros((h, w), dtype=np.uint8)))
    return mask_util.merge(rles)


def _area(rle: dict) -> float:
    return float(mask_util.area(rle))


def _intersect(rle1: dict, rle2: dict) -> dict:
    return mask_util.merge([rle1, rle2], intersect=True)


def _bbox_from_rle(rle: dict) -> Tuple[int, int, int, int]:
    bb = mask_util.toBbox(rle)  # (x,y,w,h) float
    x, y, w, h = bb
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = int(np.ceil(x + w))
    y1 = int(np.ceil(y + h))
    return x0, y0, x1, y1


def _centroid_from_rle(rle: dict) -> Tuple[float, float]:
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
    if window_source == "pred" and P is not None:
        return _centroid_from_rle(P)
    return _centroid_from_rle(G)


# ----------------------------------------------------------------------
# Intersection ratio 및 kernel 기반 score (modern path)
# ----------------------------------------------------------------------


def intersection_ratio(P: dict, G: dict, denom: str) -> float:
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
    Wmask, Warea, (x0, y0, x1, y1) = _window_mask(center_x, center_y, win_w, win_h, H, W)
    if Warea <= 0:
        return 0.0, 0.0, 0.0

    TinW = _intersect(T_mask, Wmask)
    t_area = _area(TinW)
    density = float(t_area / Warea)

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
# Legacy evaluator: labels-arr 기반 (original legacy behavior)
# ----------------------------------------------------------------------
def eval_intersection_run(
    labels_tests: Sequence[Sequence[Sequence[Dict[str, Any]]]],
    *,
    x_len: int = 20,
    y_len: int = 12,
    width: Optional[int] = None,
    height: Optional[int] = None,
    grid_w: Optional[int] = None,
    grid_h: Optional[int] = None,
    max_x: float = 3456.0,
    max_y: float = 3720.0,
    missing_as_corner: bool = False,
    **kwargs: Any,
) -> Tuple[List[ImageIR], Dict[str, float]]:
    """
    Legacy evaluation that mirrors the old `eval(...)` script.
    labels_tests[test_idx][agent_idx][frame_idx] => dict with 'vpx','vpy'
    Agent 0 is the predictor to evaluate; agents[1:] are references.
    Returns per_image list and aggregates dict (same keys as modern).

    A predictor frame marked `missing` (or carrying the -9999 sentinel) is a
    frame the model emitted nothing for, and scores exactly 0. Before this was
    handled here, the sentinel went through the clip below and was scored as a
    viewport at the map's top-left corner, which is not 0 whenever a spectator
    was watching that corner; `missing_as_corner=True` reproduces that.
    """
    if width is None:
        width = grid_w if grid_w is not None else 128
    if height is None:
        height = grid_h if grid_h is not None else 128
    total_intersection = []
    intersection_multi = []
    is_intersect = []
    is_intersect_30 = []
    is_intersect_50 = []
    per_image: List[ImageIR] = []
    synthetic_img_id = 1

    total_tests = len(labels_tests)
    step_interval = max(5000, total_tests // 10)

    for test_idx, agents in enumerate(labels_tests):
        if (test_idx + 1) % step_interval == 0 or (test_idx + 1) == total_tests:
            print(f"  [IC Eval] {test_idx + 1}/{total_tests} frames ({(test_idx + 1) / total_tests * 100:.1f}%)", flush=True)

        if not agents or len(agents) == 0:
            continue
        lengths = [len(a) for a in agents if a is not None]
        if len(lengths) == 0:
            continue
        min_length = min(lengths)

        temp_intersect = []
        for i in range(min_length):
            total_tiles = np.zeros((width, height), dtype=np.int32)

            # accumulate reference agent windows into integer count grid
            for ref_agent in agents[1:]:
                try:
                    frame = ref_agent[i]
                except Exception:
                    continue
                try:
                    vx = float(frame["vpx"])
                    vy = float(frame["vpy"])
                except Exception:
                    continue
                px = int(np.round(vx / max_x * (width - x_len)))
                py = int(np.round(vy / max_y * (height - y_len)))
                px = np.clip(px, 0, width - x_len)
                py = np.clip(py, 0, height - y_len)
                total_tiles[px : px + x_len, py : py + y_len] += 1

            # predictor
            try:
                pred_frame = agents[0][i]
                pvx = float(pred_frame["vpx"])
                pvy = float(pred_frame["vpy"])
            except Exception:
                synthetic_img_id += 1
                continue

            missing = bool(pred_frame.get("missing", False)) or (pvx <= -9999.0 and pvy <= -9999.0)
            if missing and not missing_as_corner:
                intersect_multi = 0.0
                intersection = 0.0
                pred_tiles = np.zeros((0, 0), dtype=np.int32)
            else:
                pred_x = int(np.round(pvx / max_x * (width - x_len)))
                pred_y = int(np.round(pvy / max_y * (height - y_len)))
                pred_x = np.clip(pred_x, 0, width - x_len)
                pred_y = np.clip(pred_y, 0, height - y_len)

                pred_tiles = total_tiles[pred_x : pred_x + x_len, pred_y : pred_y + y_len]

                intersect_multi = float(pred_tiles.mean())
                binary = (pred_tiles > 0).astype(np.float32)
                intersection = float(binary.mean())

            # KBRS-like centeredness/mixture using gaussian weighting on binary

            intersection_multi.append(intersect_multi)
            total_intersection.append(intersection)
            temp_intersect.append(intersection)
            is_intersect.append(1 if intersection != 0.0 else 0)
            is_intersect_30.append(1 if intersection >= 0.3 else 0)
            is_intersect_50.append(1 if intersection >= 0.5 else 0)

            # overlap_count: number of non-zero pixels in window (legacy-style approximate)
            overlap_count = int(np.sum(pred_tiles > 0))

            per_image.append(
                ImageIR(
                    image_id=synthetic_img_id,
                    width=width,
                    height=height,
                    ir=intersection,
                    overlap_count=overlap_count,
                )
            )

            synthetic_img_id += 1

        safe_mean = float(np.nanmean(temp_intersect)) if len(temp_intersect) else 0.0

    total_intersection = [0 if (x != x) else x for x in total_intersection]

    ir_values = np.array([x.ir for x in per_image], dtype=float) if per_image else np.array([])
    mean_ir = float(np.mean(ir_values)) if ir_values.size > 0 else 0.0
    median_ir = float(np.median(ir_values)) if ir_values.size > 0 else 0.0
    p90_ir = float(np.percentile(ir_values, 90)) if ir_values.size > 0 else 0.0
    coverage_any = float(np.mean(ir_values > 0.0)) if ir_values.size > 0 else 0.0
    multi_cov = float(np.mean(np.array([x.overlap_count for x in per_image]) >= 2)) if per_image else 0.0
    multi_inter_mean = float(np.mean(intersection_multi)) if len(intersection_multi) > 0 else 0.0

    coverage_th03 = float(np.mean(is_intersect_30)) if len(is_intersect_30) > 0 else (float(np.mean(ir_values >= 0.30)) if ir_values.size > 0 else 0.0)
    coverage_th05 = float(np.mean(is_intersect_50)) if len(is_intersect_50) > 0 else (float(np.mean(ir_values >= 0.50)) if ir_values.size > 0 else 0.0)

    aggregates: Dict[str, float] = {
        "mean_ir": mean_ir,
        "median_ir": median_ir,
        "p90_ir": p90_ir,
        "coverage_any": coverage_any,
        "coverage_th03": coverage_th03,
        "coverage_th05": coverage_th05,
        "multi_coverage": multi_cov,
        "multi_intersection": multi_inter_mean,
        "num_images": int(len(per_image)),
    }
    return per_image, aggregates


# alias for compatibility
def eval_run(*args, **kwargs):
    return eval_intersection_run(*args, **kwargs)
