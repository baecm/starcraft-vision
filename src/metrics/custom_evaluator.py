"""
Single-region intersection ratio (IR) and Intersection@{any, 0.3, 0.5}.

eval_intersection_run is the evaluator behind evaluate.py's per-replay IR
and ic@ numbers (the single-region tables of the thesis and the KBRS paper).
It reproduces the original evaluation script, including its conventions:
positions are top-left corners in map pixels scaled to the tile grid
(max_x / max_y), and the window is x_len x y_len tiles. Its flags re-enable
the faults fixed in 2026-08..10 (see thesis/tog-revision-notes.md), which is
how the originally reported numbers are reproduced exactly.

intersection_ratio is the mask-based definition that
metrics.modes.coverage_of reimplements with array slices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pycocotools.mask as mask_util


@dataclass
class ImageIR:
    """Per-frame result of eval_intersection_run."""
    image_id: int
    width: int
    height: int
    ir: float  # intersection ratio
    overlap_count: int  # nonzero tiles in the predicted window (an approximation kept from the original script)


def _area(rle: dict) -> float:
    return float(mask_util.area(rle))


def _intersect(rle1: dict, rle2: dict) -> dict:
    return mask_util.merge([rle1, rle2], intersect=True)


def intersection_ratio(P: dict, G: dict, denom: str) -> float:
    """|P & G| / |denominator| for two RLE masks; denom is 'gt', 'pred' or 'union'."""
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


# ----------------------------------------------------------------------
# The IR evaluator (a port of the original evaluation script)
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


