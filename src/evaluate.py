"""
Replay-level evaluation of stored predictions (make evaluate / make estimate).

    python src/evaluate.py --mode model --model-name <run> --epoch 30 --replays 275 1725 ...

For each replay, and each score threshold of --score-thresholds, one CSV row
with
  single-region   IR and Intersection@{any, 0.3, 0.5} of the top prediction
                  (compute_ic_for_replay -> metrics.custom_evaluator), and the
                  KBRS cue scores at the predicted windows (compute_kbrs_for_replay)
  multi-region    CWO, M-CTI, jerk, jump rate, event recall, pairwise overlap
                  (compute_multi_region_for_replay -> metrics.evaluator)
--task single|multi|all picks which. --mode gt scores the observers
themselves instead of a model, and --mode feature computes the KBRS cue maps
over the whole input instead of at predicted windows.

CSV columns and the names the papers use for them:
  ic_ratio    IR, the mean intersection ratio of the top prediction
  ic@000      Intersection@any  (fraction of frames with IR > 0)
  ic@030      Intersection@0.3
  ic@050      Intersection@0.5
  ic_multi    mean number of observers covering a tile of the predicted window
  missing_preds, evaluated_frames
              frames with no prediction above the threshold (scored 0, i.e.
              coverage-penalized), and frames scored
The column names are kept as they are because existing result CSVs use them.

The flags --skip-missing-preds, --legacy-centroid, --missing-as-corner and
--legacy-anchor re-enable the evaluator faults fixed in 2026-08..10, which
reproduces the originally reported numbers exactly (thesis/tog-revision-notes.md).

A second, older mode scores prediction files directly (--gt/--gt-dir with
--pred/--pred-dir; run_kernel_eval).
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
from typing import Any, Dict, List, Sequence, Tuple, Optional, Union
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from metrics import eval_intersection_run, ImageIR
from metrics.evaluator import MultiRegionEvaluator
from utils.report import ReportBlock, print_section_header

# Ensure unbuffered stdout in container/redirection environments
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# The IR evaluator works in the coordinates of the original evaluation script:
# a camera position is the screen's top-left corner in map pixels, from 0 to
# MAXCOORD. max_x = 3456 is the 4096 px map less the 640 px screen; max_y is
# the original script's value. Positions on the tile grid are scaled into it.
IC_KERNEL_WH = (20, 12)           # viewport (width, height) in tiles
IC_GRID_WH = (128, 128)           # tile map (width, height)
IC_MAXCOORD_XY = (3456.0, 3720.0)  # largest top-left position, map pixels


def _pair(value: Any, default: Tuple, cast) -> Tuple:
    """A (w, h)-style pair from "20,12", a 2-sequence, or None (the default)."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return cast(value[0]), cast(value[1])
    if isinstance(value, str) and "," in value:
        a, b = value.split(",")
        return cast(a), cast(b)
    raise ValueError(f"expected 'a,b' or a pair, got {value!r}")


def _nan_ic_metrics() -> Dict[str, float]:
    """The IR / Intersection@ columns of a row with nothing to score."""
    return {
        "ic@000": float("nan"),
        "ic@030": float("nan"),
        "ic@050": float("nan"),
        "ic_multi": float("nan"),
        "ic_ratio": float("nan"),
        "median_ir": float("nan"),
        "p90_ir": float("nan"),
    }


# =====================================================================
# 1. COCO & Geometry Helpers
# =====================================================================

def _centroid_from_coco_ann(
    ann: dict,
    img_w: int,
    img_h: int,
    size_wh: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """
    Return centroid (cx, cy) in image pixel coordinates (0..img_w-1, 0..img_h-1)
    Handles 'segmentation' (polygon or RLE) and 'bbox'.

    `size_wh` takes the annotation's top-left corner but measures the centroid
    with that (width, height) instead of the stored one. Pass it for
    predictions: inference clamps boxes to the map, so a box hanging off an
    edge is written narrower than the viewport it denotes and its centroid
    comes out pulled toward that edge by (w_true - w_stored) / 2. Ground-truth
    annotations carry their true size, so they are measured as stored.
    """
    def _center(x, y, w, h) -> Tuple[float, float]:
        if size_wh is not None:
            w, h = size_wh
        return float(x + w / 2.0), float(y + h / 2.0)

    if "segmentation" in ann and ann["segmentation"]:
        seg = ann["segmentation"]
        if isinstance(seg, dict) and "counts" in seg:
            bbox = mask_util.toBbox(seg)
        else:
            try:
                rles = mask_util.frPyObjects(seg, img_h, img_w) if isinstance(seg, list) else seg
                mrg = mask_util.merge(rles)
                bbox = mask_util.toBbox(mrg)
            except Exception:
                bbox = None
        if bbox is not None:
            x, y, w, h = bbox
            return _center(x, y, w, h)

    if "bbox" in ann and ann["bbox"]:
        x, y, w, h = ann["bbox"]
        return _center(x, y, w, h)

    return float(img_w) / 2.0, float(img_h) / 2.0


def _topleft_from_coco_ann(ann: dict, img_w: int, img_h: int) -> Tuple[float, float]:
    """
    Return the top-left corner (x, y) of an annotation, in the same pixel
    coordinates as `_centroid_from_coco_ann`.

    This, not the centroid, is what the kernel evaluator expects. It descends
    from the original evaluation script, whose agent traces store the camera's
    top-left position in map pixels - max_x = 3456 is the 4096 px map less the
    640 px screen - and it places each window at [px, px + x_len). Feeding it a
    centroid shifts every window by half a viewport and, once the shift pushes
    a window past the far edge, the clip below stacks windows from up to half a
    viewport apart onto one position, which inflates overlap there.
    """
    # Reuse the centroid helper's bbox resolution, then undo the half-size step.
    cx, cy = _centroid_from_coco_ann(ann, img_w, img_h, size_wh=(0.0, 0.0))
    return cx, cy


def coco_to_kernel_labels(
    coco_gt: COCO,
    preds_list: Union[List[dict], Dict[int, List[dict]]],
    *,
    x_len: int,
    y_len: int,
    grid_w: int,
    grid_h: int,
    max_x: float,
    max_y: float,
    score_thresh: float = 0.0,
    skip_missing_preds: bool = False,
    legacy_centroid: bool = False,
    legacy_anchor: bool = False,
    return_stats: bool = False,
):
    """
    Convert COCO-style GT + preds into agent-trace tests for kernel-based evaluator.
    Only predictions with score >= score_thresh are considered valid.

    The evaluator places each window at the agent's position, as a top-left
    corner, so that is what is passed for both the prediction and every
    observer (see `_topleft_from_coco_ann`).

    `skip_missing_preds`, `legacy_centroid` and `legacy_anchor` reproduce the
    evaluator as it was before 8de59a3/cf46716, 0782acd and the anchor fix
    respectively, so that numbers reported from those versions can be
    re-derived and each fix measured separately. `legacy_anchor` passes
    centroids where corners are expected; `legacy_centroid` only matters with
    it, since the corrected path takes no centroid. Leave all off for the
    current, corrected behavior.
    """
    if isinstance(preds_list, dict):
        preds_by_img = preds_list
    else:
        preds_by_img = {}
        for p in preds_list:
            img_id = int(p["image_id"])
            preds_by_img.setdefault(img_id, []).append(p)

    tests: List[List[List[Dict[str, float]]]] = []
    stats = {
        "total_frames": 0,
        "missing_preds": 0,
        "no_gt_anns": 0,
        "evaluated": 0,
    }

    img_to_anns = getattr(coco_gt, "imgToAnns", None)

    for img in coco_gt.dataset.get("images", []):
        image_id = int(img["id"])
        stats["total_frames"] += 1
        img_w = int(img.get("width", grid_w))
        img_h = int(img.get("height", grid_h))

        if img_to_anns is not None:
            anns = img_to_anns.get(image_id, [])
        else:
            ann_ids = coco_gt.getAnnIds(imgIds=image_id)
            anns = coco_gt.loadAnns(ann_ids) if ann_ids else []

        img_preds = preds_by_img.get(image_id, [])
        valid_preds = [p for p in img_preds if float(p.get("score", 1.0)) >= score_thresh]

        if len(valid_preds) == 0:
            stats["missing_preds"] += 1
            if skip_missing_preds:
                continue
            dummy_vx, dummy_vy = -9999.0, -9999.0
            agent0 = [{"vpx": dummy_vx, "vpy": dummy_vy, "missing": True}]
        else:
            best_pred = max(valid_preds, key=lambda q: float(q.get("score", 0.0)))
            if legacy_anchor:
                # a prediction denotes a top-left plus the fixed viewport size; its
                # stored w/h are clamped at the map edge and would skew the centroid
                pcx, pcy = _centroid_from_coco_ann(
                    best_pred, img_w, img_h,
                    size_wh=None if legacy_centroid else (x_len, y_len),
                )
            else:
                pcx, pcy = _topleft_from_coco_ann(best_pred, img_w, img_h)
            vx = float(pcx) / max(1, (img_w - x_len)) * max_x
            vy = float(pcy) / max(1, (img_h - y_len)) * max_y
            agent0 = [{"vpx": vx, "vpy": vy}]

        ref_agents: List[List[Dict[str, float]]] = []
        for ann in anns:
            if legacy_anchor:
                gcx, gcy = _centroid_from_coco_ann(ann, img_w, img_h)
            else:
                gcx, gcy = _topleft_from_coco_ann(ann, img_w, img_h)
            gvx = float(gcx) / max(1, (img_w - x_len)) * max_x
            gvy = float(gcy) / max(1, (img_h - y_len)) * max_y
            ref_agents.append([{"vpx": gvx, "vpy": gvy}])

        test_agents: List[List[Dict[str, float]]] = [agent0] + ref_agents
        if len(ref_agents) == 0:
            stats["no_gt_anns"] += 1
            continue

        tests.append(test_agents)
        stats["evaluated"] += 1

    if return_stats:
        return tests, stats
    return tests


def eval_kernel_from_coco(
    coco_gt: COCO,
    preds_list: Union[List[dict], Dict[int, List[dict]]],
    *,
    name: str = "run",
    kernel: Tuple[int, int] = (20, 12),
    grid: Tuple[int, int] = (128, 128),
    maxcoord: Tuple[float, float] = (3456.0, 3720.0),
    skip_missing_preds: bool = False,
    score_thresh: float = 0.0,
    legacy_centroid: bool = False,
    missing_as_corner: bool = False,
    legacy_anchor: bool = False,
) -> Tuple[Dict[str, Any], List[ImageIR], Dict[str, float]]:
    """
    Evaluate kernel metrics directly from COCO objects.
    """
    x_len, y_len = kernel
    grid_w, grid_h = grid
    max_x, max_y = maxcoord

    labels_tests, stats = coco_to_kernel_labels(
        coco_gt,
        preds_list,
        x_len=x_len,
        y_len=y_len,
        grid_w=grid_w,
        grid_h=grid_h,
        max_x=max_x,
        max_y=max_y,
        score_thresh=score_thresh,
        skip_missing_preds=skip_missing_preds,
        legacy_centroid=legacy_centroid,
        legacy_anchor=legacy_anchor,
        return_stats=True,
    )

    if len(labels_tests) == 0:
        empty_row = {
            "name": name,
            "score_thresh": float(score_thresh),
            "kernel": f"{x_len}x{y_len}",
            "num_images": 0,
            **_nan_ic_metrics(),
            "total_frames": stats.get("total_frames", 0),
            "missing_preds": stats.get("missing_preds", 0),
            "no_gt_anns": stats.get("no_gt_anns", 0),
            "evaluated_frames": 0,
        }
        return empty_row, [], {}

    ir_per_image, agg = eval_intersection_run(
        labels_tests,
        x_len=x_len,
        y_len=y_len,
        width=grid_w,
        height=grid_h,
        max_x=max_x,
        max_y=max_y,
        missing_as_corner=missing_as_corner,
    )

    ir_values = np.array([x.ir for x in ir_per_image], dtype=float) if ir_per_image else np.array([])
    ic000 = float(agg.get("coverage_any", np.mean(ir_values > 0.0) if ir_values.size > 0 else 0.0))
    ic030 = float(agg.get("coverage_th03", np.mean(ir_values >= 0.30) if ir_values.size > 0 else 0.0))
    ic050 = float(agg.get("coverage_th05", np.mean(ir_values >= 0.50) if ir_values.size > 0 else 0.0))
    ic_multi = float(agg.get("multi_intersection", agg.get("multi_coverage", 0.0)))
    mean_ir = float(agg.get("mean_ir", np.mean(ir_values) if ir_values.size > 0 else 0.0))
    median_ir = float(agg.get("median_ir", np.median(ir_values) if ir_values.size > 0 else 0.0))
    p90_ir = float(agg.get("p90_ir", np.percentile(ir_values, 90) if ir_values.size > 0 else 0.0))

    row = {
        "name": name,
        "score_thresh": float(score_thresh),
        "kernel": f"{x_len}x{y_len}",
        "num_images": int(agg.get("num_images", len(ir_per_image))),
        "ic@000": ic000,
        "ic@030": ic030,
        "ic@050": ic050,
        "ic_multi": ic_multi,
        "ic_ratio": mean_ir,
        "median_ir": median_ir,
        "p90_ir": p90_ir,
        "total_frames": int(stats.get("total_frames", 0)),
        "missing_preds": int(stats.get("missing_preds", 0)),
        "no_gt_anns": int(stats.get("no_gt_anns", 0)),
        "evaluated_frames": int(stats.get("evaluated", len(ir_per_image))),
    }

    for extra_k in [
        "top1_ic@050", "cluster_recall@top1", "min_gt_dist_top1",
        "top3_ic@050", "cluster_recall@top3", "min_gt_dist_top3",
        "top5_ic@050", "cluster_recall@top5", "min_gt_dist_top5",
        "prec_r10", "recall_r10", "prec_r20", "recall_r20",
        "inter_region_dist", "center_jitter", "target_persistence",
    ]:
        if extra_k in agg:
            row[extra_k] = float(agg[extra_k])

    return row, ir_per_image, agg


# =====================================================================
# 2. KBRS cue scores at predicted windows (density / centeredness / mixture)
# =====================================================================
# A diagnostic, not the training-time KBRS of models/plugins/kbrs: it reads
# the raw input frame (.npy), takes the maximum over all channels, binarizes
# it at --threshold, and scores a window of --window tiles centered on each
# prediction (or each observer, --mode gt):
#   density       fraction of occupied tiles in the window
#   centeredness  Gaussian-weighted (sigma = h/4) occupancy, clipped to [0, 1]
#   mixture       density x centeredness
# Because every channel counts, any channel that covers the whole map makes
# every window fully occupied and all three scores 1.

def make_gaussian_kernel(h: int, w: int, sigma: Optional[float] = None) -> np.ndarray:
    if sigma is None:
        sigma = h / 4.0
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    dist2 = (ys - cy) ** 2 + (xs - cx) ** 2
    g = np.exp(-dist2 / (2.0 * sigma * sigma))
    return g.astype(np.float32)


def kbrs_scores_from_patch(
    patch: np.ndarray,
    threshold: float = 0.0,
) -> Tuple[float, float, float]:
    if patch.ndim == 3:
        patch2d = patch.max(axis=0)
    elif patch.ndim == 2:
        patch2d = patch
    else:
        raise ValueError(f"Unsupported patch shape: {patch.shape}")

    if patch2d.size == 0:
        return 0.0, 0.0, 0.0

    binary = (patch2d > threshold).astype(np.float32)
    density = float(binary.mean())

    h, w = binary.shape
    g = make_gaussian_kernel(h, w, sigma=h / 4.0)
    g_sum = float(g.sum()) + 1e-12

    centered_raw = float((binary * g).sum() / g_sum)
    centeredness = float(max(0.0, min(1.0, centered_raw)))
    mixture = density * centeredness
    return density, centeredness, mixture


def extract_window(
    feat: np.ndarray,
    center_x: float,
    center_y: float,
    win_w: int,
    win_h: int,
) -> np.ndarray:
    if feat.ndim == 3:
        C, H, W = feat.shape
    elif feat.ndim == 2:
        H, W = feat.shape
    else:
        raise ValueError(f"Unsupported feature shape: {feat.shape}")

    half_w = win_w // 2
    half_h = win_h // 2
    x0 = max(0, int(round(center_x)) - half_w)
    y0 = max(0, int(round(center_y)) - half_h)
    x1 = min(W, x0 + win_w)
    y1 = min(H, y0 + win_h)

    if x1 <= x0 or y1 <= y0:
        return feat[:, 0:0, 0:0] if feat.ndim == 3 else feat[0:0, 0:0]

    return feat[:, y0:y1, x0:x1] if feat.ndim == 3 else feat[y0:y1, x0:x1]


def compute_kbrs_grid(
    feat: np.ndarray,
    win_w: int,
    win_h: int,
    stride_x: int = 1,
    stride_y: int = 1,
    threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if feat.ndim == 3:
        C, H, W = feat.shape
    elif feat.ndim == 2:
        H, W = feat.shape
    else:
        raise ValueError(f"Unsupported feature shape: {feat.shape}")

    out_h = max(0, (H - win_h) // stride_y + 1)
    out_w = max(0, (W - win_w) // stride_x + 1)

    density_map = np.zeros((out_h, out_w), dtype=np.float32)
    centeredness_map = np.zeros((out_h, out_w), dtype=np.float32)
    mixture_map = np.zeros((out_h, out_w), dtype=np.float32)

    for oy in range(out_h):
        for ox in range(out_w):
            cx = ox * stride_x + win_w / 2.0
            cy = oy * stride_y + win_h / 2.0
            patch = extract_window(feat, cx, cy, win_w, win_h)
            d, c, m = kbrs_scores_from_patch(patch, threshold=threshold)
            density_map[oy, ox] = d
            centeredness_map[oy, ox] = c
            mixture_map[oy, ox] = m

    return density_map, centeredness_map, mixture_map


def build_feature_path_from_meta(
    input_root: str,
    replay_id: str,
    file_name: str,
    img_id: int,
    use_file_name: bool,
    feature_ext: str,
) -> str:
    rep_dir = os.path.join(input_root, f"{replay_id}.rep")
    if use_file_name and file_name:
        base, _ext = os.path.splitext(file_name)
        fname = base + feature_ext
    else:
        fname = f"{img_id}{feature_ext}"
    return os.path.join(rep_dir, fname)


def _process_image_worker(args: Tuple) -> Optional[Dict]:
    (
        mode,
        replay_id,
        img_id,
        file_name,
        img_w,
        img_h,
        input_root,
        feature_ext,
        use_file_name,
        win_w,
        win_h,
        stride_x,
        stride_y,
        threshold,
        positions,
        source_tag,
    ) = args

    feat_path = build_feature_path_from_meta(
        input_root=input_root,
        replay_id=replay_id,
        file_name=file_name,
        img_id=img_id,
        use_file_name=use_file_name,
        feature_ext=feature_ext,
    )

    if not os.path.isfile(feat_path):
        return None

    try:
        feat = np.load(feat_path)
    except Exception as e:
        print(f"[Error][replay={replay_id}] failed to load feature {feat_path}: {e}")
        return None

    if mode == "feature":
        density_map, centeredness_map, mixture_map = compute_kbrs_grid(
            feat,
            win_w=win_w,
            win_h=win_h,
            stride_x=stride_x,
            stride_y=stride_y,
            threshold=threshold,
        )
        return {
            "replay_id": replay_id,
            "image_id": img_id,
            "file_name": file_name,
            "source": source_tag,
            "num_points": 0,
            "mean_density": float(density_map.mean()),
            "max_density": float(density_map.max()),
            "mean_centeredness": float(centeredness_map.mean()),
            "max_centeredness": float(centeredness_map.max()),
            "mean_mixture": float(mixture_map.mean()),
            "max_mixture": float(mixture_map.max()),
        }

    if not positions:
        return None

    ds, cs, ms = [], [], []
    for cx, cy in positions:
        patch = extract_window(feat, cx, cy, win_w, win_h)
        d, c, m = kbrs_scores_from_patch(patch, threshold=threshold)
        ds.append(d)
        cs.append(c)
        ms.append(m)

    if not ds:
        return None

    ds_arr = np.array(ds, dtype=np.float32)
    cs_arr = np.array(cs, dtype=np.float32)
    ms_arr = np.array(ms, dtype=np.float32)

    return {
        "replay_id": replay_id,
        "image_id": img_id,
        "file_name": file_name,
        "source": source_tag,
        "num_points": int(len(ds)),
        "mean_density": float(ds_arr.mean()),
        "max_density": float(ds_arr.max()),
        "mean_centeredness": float(cs_arr.mean()),
        "max_centeredness": float(cs_arr.max()),
        "mean_mixture": float(ms_arr.mean()),
        "max_mixture": float(ms_arr.max()),
    }


def compute_kbrs_for_replay(
    replay_id: str,
    mode: str,
    coco_gt: COCO,
    args: argparse.Namespace,
    preds_by_img: Optional[Dict[int, List[dict]]] = None,
    model_tag: Optional[str] = None,
) -> Tuple[float, float, float, int]:
    """Mean KBRS cue scores of one replay (see the section comment), and the
    number of frames scored."""
    win_w, win_h = _pair(getattr(args, "window", None), IC_KERNEL_WH, int)
    stride_x, stride_y = _pair(getattr(args, "stride", None), (1, 1), int)

    feature_ext = getattr(args, "feature_ext", ".npy")
    use_file_name = getattr(args, "use_file_name", False)
    threshold = getattr(args, "threshold", 0.0)
    max_frames = getattr(args, "max_frames", -1)

    images = list(coco_gt.dataset.get("images", []))
    if max_frames > 0:
        images = images[:max_frames]

    tasks: List[Tuple] = []
    for img in images:
        img_id = int(img["id"])
        file_name = img.get("file_name", "")
        img_w = int(img.get("width", 0))
        img_h = int(img.get("height", 0))

        if mode == "feature":
            positions = None
            source_tag = "feature"
        elif mode == "gt":
            ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
            anns = coco_gt.loadAnns(ann_ids) if ann_ids else []
            if not anns:
                continue
            positions = [_centroid_from_coco_ann(ann, img_w, img_h) for ann in anns]
            if not positions:
                continue
            source_tag = "gt"
        else:
            if not preds_by_img:
                continue
            img_preds = preds_by_img.get(img_id, [])
            if not img_preds:
                continue
            positions = [
                _centroid_from_coco_ann(det, img_w, img_h, size_wh=(win_w, win_h))
                for det in img_preds
                if "bbox" in det and det["bbox"] is not None
            ]
            if not positions:
                continue
            source_tag = f"model:{model_tag}"

        tasks.append(
            (
                mode,
                replay_id,
                img_id,
                file_name,
                img_w,
                img_h,
                args.input_root,
                feature_ext,
                use_file_name,
                win_w,
                win_h,
                stride_x,
                stride_y,
                threshold,
                positions,
                source_tag,
            )
        )

    if not tasks:
        return float("nan"), float("nan"), float("nan"), 0

    rows: List[Dict] = []
    num_workers = getattr(args, "num_workers", 0)
    if num_workers == 1:
        for t in tasks:
            row = _process_image_worker(t)
            if row is not None:
                rows.append(row)
    else:
        n_workers = num_workers or cpu_count()
        with Pool(processes=n_workers) as pool:
            for row in pool.imap_unordered(_process_image_worker, tasks):
                if row is not None:
                    rows.append(row)

    if not rows:
        return float("nan"), float("nan"), float("nan"), 0

    df = pd.DataFrame(rows)
    mean_density = float(df["mean_density"].mean())
    mean_centered = float(df["mean_centeredness"].mean())
    mean_mixture = float(df["mean_mixture"].mean())
    num_images = int(df["image_id"].nunique())

    print(
        f"[KBRS replay={replay_id}] num_images={num_images}, "
        f"mean_density={mean_density:.4f}, mean_centeredness={mean_centered:.4f}, mean_mixture={mean_mixture:.4f}"
    )
    return mean_density, mean_centered, mean_mixture, num_images


# =====================================================================
# 3. IC & Multi-Region Calculation per Replay
# =====================================================================

def load_coco_gt(label_root: str, replay_id: str, label_method: str) -> COCO:
    gt_dir = os.path.join(label_root, f"{replay_id}.rep")
    gt_path = os.path.join(gt_dir, f"{label_method}.json")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Ground truth (COCO) file not found: {gt_path}")
    return COCO(gt_path)


def load_coco_preds(
    pred_root: str,
    model_name: str,
    epoch: int,
    replay_id: str,
    label_method: str,
    score_threshold=None,
) -> Dict[int, List[dict]]:
    # Inference writes a threshold's predictions to their own directory,
    # `model_NNN_th<x>`, and leaves `model_NNN` for a run that set none. Naming
    # the threshold here is what lets a sweep be read back: the region count a
    # proposal detector emits is set by this filter, so comparing two methods
    # at a matched count means addressing those directories separately.
    suffix = "" if score_threshold is None else f"_th{score_threshold}"
    pred_dir = os.path.join(
        pred_root, model_name, f"model_{epoch:03d}{suffix}", f"{replay_id}.rep"
    )
    pred_path = os.path.join(pred_dir, f"{label_method}.json")
    if not os.path.isfile(pred_path):
        print(f"[!] Warning: Prediction file not found: {pred_path} -> Proceeding with fallback/empty detections.", flush=True)
        return {}

    file_size_mb = os.path.getsize(pred_path) / (1024 * 1024)
    print(f"[INFO] Loading predictions from {pred_path} ({file_size_mb:.1f} MB)...", flush=True)
    t0 = time.time()
    with open(pred_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    if isinstance(loaded, dict) and "annotations" in loaded:
        dets = loaded["annotations"]
    elif isinstance(loaded, list):
        dets = loaded
    else:
        raise ValueError("Unsupported prediction JSON format for COCO dets.")

    preds_by_img: Dict[int, List[dict]] = {}
    for det in dets:
        img_id = int(det["image_id"])
        preds_by_img.setdefault(img_id, []).append(det)

    elapsed = time.time() - t0
    print(f"[INFO] Successfully loaded {len(dets)} predictions across {len(preds_by_img)} frames in {elapsed:.2f}s", flush=True)
    return preds_by_img


def compute_ic_for_replay(
    replay_id: str,
    mode: str,
    coco_gt: COCO,
    args: argparse.Namespace,
    preds_all: Optional[Union[List[dict], Dict[int, List[dict]]]] = None,
    model_tag: Optional[str] = None,
    skip_missing_preds: bool = False,
    score_thresh: float = 0.0,
    legacy_centroid: bool = False,
    missing_as_corner: bool = False,
    legacy_anchor: bool = False,
) -> Dict[str, float]:
    """
    Computes IC metrics for a replay sequence with a specified score threshold.
    """
    ic_x_len, ic_y_len = _pair(getattr(args, "ic_kernel", None), IC_KERNEL_WH, int)
    ic_grid_w, ic_grid_h = _pair(getattr(args, "ic_grid", None), IC_GRID_WH, int)
    ic_max_x, ic_max_y = _pair(getattr(args, "ic_maxcoord", None), IC_MAXCOORD_XY, float)

    images = list(coco_gt.dataset.get("images", []))
    max_frames = getattr(args, "max_frames", -1)
    if max_frames > 0:
        images = images[:max_frames]

    num_images = len(images)
    num_preds = sum(len(v) for v in preds_all.values()) if isinstance(preds_all, dict) else (len(preds_all) if preds_all else 0)

    print(
        f"[IC replay={replay_id}] start "
        f"(mode={mode}, thresh={score_thresh}, kernel={ic_x_len}x{ic_y_len}, "
        f"grid={ic_grid_w}x{ic_grid_h}, maxcoord=({ic_max_x},{ic_max_y}), "
        f"num_images={num_images}, num_preds={num_preds}, skip_missing={skip_missing_preds})"
    )
    # returned when there is nothing to score (no observer boxes, or no predictions)
    empty_row = {
        "kernel": f"{ic_x_len}x{ic_y_len}",
        "score_thresh": float(score_thresh),
        "num_images": num_images,
        **_nan_ic_metrics(),
    }

    if mode == "gt":
        gt_dets: List[dict] = []
        for img in images:
            img_id = int(img["id"])
            ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
            anns = coco_gt.loadAnns(ann_ids) if ann_ids else []
            for ann in anns:
                bbox = ann.get("bbox", None)
                if bbox is None:
                    continue
                gt_dets.append({
                    "image_id": img_id,
                    "bbox": bbox,
                    "score": 1.0,
                    "category_id": int(ann.get("category_id", 1)),
                })

        if gt_dets:
            tag = f"gt_r{replay_id}"
            row, per_img_ic, agg_ic = eval_kernel_from_coco(
                coco_gt,
                gt_dets,
                name=tag,
                kernel=(ic_x_len, ic_y_len),
                grid=(ic_grid_w, ic_grid_h),
                maxcoord=(ic_max_x, ic_max_y),
                skip_missing_preds=skip_missing_preds,
                score_thresh=score_thresh,
            )
            return row

        return empty_row

    if mode == "model" and preds_all:
        t_ic = time.time()
        row, per_img_ic, agg_ic = eval_kernel_from_coco(
            coco_gt,
            preds_all,
            name=f"{model_tag}_r{replay_id}",
            kernel=(ic_x_len, ic_y_len),
            grid=(ic_grid_w, ic_grid_h),
            maxcoord=(ic_max_x, ic_max_y),
            skip_missing_preds=skip_missing_preds,
            score_thresh=score_thresh,
            legacy_centroid=legacy_centroid,
            missing_as_corner=missing_as_corner,
            legacy_anchor=legacy_anchor,
        )
        print(
            f"[IC replay={replay_id}] thresh={score_thresh} done (t={time.time() - t_ic:.2f}s) "
            f"-> IC@0.0={row.get('ic@000', float('nan')):.3f}, IC@0.5={row.get('ic@050', float('nan')):.3f}, IC_multi={row.get('ic_multi', float('nan')):.3f}",
            flush=True,
        )
        return row

    return empty_row


NAN_MULTI_ROW = {
    "cwo": float("nan"),
    "m_cti": float("nan"),
    "jerk": float("nan"),
    "jump_rate": float("nan"),
    "event_recall": float("nan"),
    "pairwise_overlap": float("nan"),
}

# A frame with no prediction still needs a viewport for the sequence metrics
# to stay aligned frame by frame; it is given this box, a 32x32-tile square at
# the map corner (xyxy, tiles). It enters CWO and M-CTI as a real viewport.
EMPTY_FRAME_BOX = [0.0, 0.0, 32.0, 32.0]


def compute_multi_region_for_replay(
    replay_id: str,
    coco_gt: COCO,
    preds_by_img: Optional[Dict[int, List[dict]]],
    grid_size: Tuple[int, int] = IC_GRID_WH,
    multi_topk: int = 3,
) -> Dict[str, float]:
    """CWO, M-CTI (with jerk and jump rate), event recall and pairwise overlap
    of the top-`multi_topk` predictions over one replay (metrics.evaluator)."""
    images = list(coco_gt.dataset.get("images", []))
    if not images:
        return dict(NAN_MULTI_ROW)

    evaluator = MultiRegionEvaluator(
        grid_size=grid_size,
        jump_threshold=35.0,
        box_format="xyxy",
        m_cti_weights=(0.4, 0.4, 0.2),
    )

    viewports_seq = []
    consensus_maps_seq = []
    event_coords_seq = []
    grid_w, grid_h = grid_size
    num_images = len(images)

    topk_label = f"top-{multi_topk}" if multi_topk > 0 else "all"
    print(f"[Multi-Region replay={replay_id}] Preparing sequences ({num_images} frames, viewports={topk_label})...", flush=True)
    t0 = time.time()
    step_interval = max(5000, num_images // 10)
    img_to_anns = getattr(coco_gt, "imgToAnns", None)

    for idx, img in enumerate(images):
        if (idx + 1) % step_interval == 0 or (idx + 1) == num_images:
            pct = (idx + 1) / num_images * 100.0
            print(f"  [Multi-Region Prep] {idx + 1}/{num_images} frames ({pct:.1f}%)", flush=True)

        img_id = int(img["id"])
        h_img = float(img.get("height", grid_h))
        w_img = float(img.get("width", grid_w))

        img_boxes = []
        if preds_by_img and img_id in preds_by_img:
            dets = preds_by_img[img_id]
            if multi_topk > 0 and len(dets) > multi_topk:
                dets = sorted(dets, key=lambda d: float(d.get("score", 0.0)), reverse=True)[:multi_topk]
            for det in dets:
                b = det.get("bbox", None)
                if b is not None and len(b) == 4:
                    x, y, w, h = b
                    x1 = (x / w_img) * grid_w
                    y1 = (y / h_img) * grid_h
                    x2 = ((x + w) / w_img) * grid_w
                    y2 = ((y + h) / h_img) * grid_h
                    img_boxes.append([x1, y1, x2, y2])

        if not img_boxes:
            img_boxes = [list(EMPTY_FRAME_BOX)]
        viewports_seq.append(np.array(img_boxes, dtype=np.float32))

        if img_to_anns is not None:
            anns = img_to_anns.get(img_id, [])
        else:
            ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
            anns = coco_gt.loadAnns(ann_ids) if ann_ids else []

        c_map = np.zeros((grid_h, grid_w), dtype=np.float32)
        events = []
        for ann in anns:
            bbox = ann.get("bbox", None)
            if bbox is not None and len(bbox) == 4:
                x, y, w, h = bbox
                cx = (x + w / 2.0) / w_img * grid_w
                cy = (y + h / 2.0) / h_img * grid_h
                events.append([cx, cy])

                x1_i = int(np.clip((x / w_img) * grid_w, 0, grid_w))
                x2_i = int(np.clip(((x + w) / w_img) * grid_w, 0, grid_w))
                y1_i = int(np.clip((y / h_img) * grid_h, 0, grid_h))
                y2_i = int(np.clip(((y + h) / h_img) * grid_h, 0, grid_h))
                if x2_i > x1_i and y2_i > y1_i:
                    c_map[y1_i:y2_i, x1_i:x2_i] += 1.0

        if c_map.max() > 0:
            c_map /= c_map.max()

        consensus_maps_seq.append(c_map)
        event_coords_seq.append(np.array(events, dtype=np.float32) if events else np.empty((0, 2), dtype=np.float32))

    print(f"[Multi-Region replay={replay_id}] Evaluating 3D metrics (CWO, Event Recall, Pairwise Overlap, M-CTI)...", flush=True)
    t_eval = time.time()
    metrics = evaluator.evaluate_sequence(
        viewports_seq=viewports_seq,
        consensus_maps_seq=consensus_maps_seq,
        event_coords_seq=event_coords_seq,
    )
    print(
        f"[Multi-Region replay={replay_id}] Done (prep={t_eval - t0:.1f}s, eval={time.time() - t_eval:.1f}s) "
        f"-> CWO={metrics.get('cwo', 0.0):.4f}, M-CTI={metrics.get('m_cti', 0.0):.4f}, Recall={metrics.get('event_recall', 0.0):.4f}, Overlap={metrics.get('pairwise_overlap', 0.0):.4f}",
        flush=True,
    )
    return metrics


# =====================================================================
# 4. File-to-file mode: score prediction files against one GT file
# =====================================================================

def run_kernel_eval(
    gt_path: Optional[str] = None,
    gt_dir: Optional[str] = None,
    pred_files: Optional[Sequence[str]] = None,
    pred_dir: Optional[str] = None,
    out_dir: str = "./results",
    names: Optional[Sequence[str]] = None,
    kernel: Optional[str] = None,
    grid: Optional[str] = None,
    maxcoord: Optional[str] = None,
    score_thresholds: Sequence[float] = (0.0,),
    run_tag: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str, str]:
    """IR / Intersection@ of each prediction file at each score threshold, against
    gt_path (or the first .json in gt_dir). Writes summary[_<run_tag>].csv/.json
    to out_dir and returns (rows, summary, csv path, json path)."""
    if gt_path is None:
        if gt_dir is None:
            raise ValueError("Either gt_path or gt_dir must be provided")
        files = [f for f in os.listdir(gt_dir) if f.endswith(".json")]
        if not files:
            raise FileNotFoundError(f"No json files found in {gt_dir}")
        gt_path = os.path.join(gt_dir, files[0])

    coco_gt = COCO(gt_path)

    preds_to_eval: List[str] = []
    if pred_files:
        preds_to_eval.extend(pred_files)
    elif pred_dir:
        for root, _, files in os.walk(pred_dir):
            for f in files:
                if f.endswith(".json"):
                    preds_to_eval.append(os.path.join(root, f))
    else:
        raise ValueError("Either pred_files or pred_dir must be provided")

    x_len, y_len = _pair(kernel, IC_KERNEL_WH, int)
    grid_w, grid_h = _pair(grid, IC_GRID_WH, int)
    max_x, max_y = _pair(maxcoord, IC_MAXCOORD_XY, float)

    all_rows: List[Dict[str, Any]] = []
    summary_json: Dict[str, Any] = {}

    for idx, pf in enumerate(preds_to_eval):
        name = names[idx] if names and idx < len(names) else os.path.splitext(os.path.basename(pf))[0]
        with open(pf, "r", encoding="utf-8") as f:
            preds_loaded = json.load(f)
        if isinstance(preds_loaded, dict) and "annotations" in preds_loaded:
            preds_list = preds_loaded["annotations"]
        elif isinstance(preds_loaded, list):
            preds_list = preds_loaded
        else:
            raise ValueError(f"Unrecognized prediction format in {pf}")

        for th in score_thresholds:
            run_name = f"{name}_th{th:.2f}"
            row, ir_per_image, agg = eval_kernel_from_coco(
                coco_gt,
                preds_list,
                name=run_name,
                kernel=(x_len, y_len),
                grid=(grid_w, grid_h),
                maxcoord=(max_x, max_y),
                score_thresh=th,
            )
            all_rows.append(row)
            summary_json[run_name] = agg

    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{run_tag}" if run_tag else ""
    csv_path = os.path.join(out_dir, f"summary{suffix}.csv")
    json_path = os.path.join(out_dir, f"summary{suffix}.json")

    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    return all_rows, summary_json, csv_path, json_path


# =====================================================================
# 5. CLI Parser & Main Entry Point
# =====================================================================

def parse_thresholds(raw_thresholds: Any) -> List[float]:
    """
    Parses score thresholds from various input formats:
      - Space-separated: ['0.0', '0.1', '0.2']
      - Comma-separated string: '0.0,0.1,0.2'
      - Mixed: ['0.0,0.1', '0.2,0.3']
    """
    if raw_thresholds is None:
        return [0.0]

    if isinstance(raw_thresholds, (int, float)):
        return [float(raw_thresholds)]

    if isinstance(raw_thresholds, str):
        parts = [p.strip() for p in raw_thresholds.split(",") if p.strip()]
        return [float(p) for p in parts] if parts else [0.0]

    results: List[float] = []
    for item in raw_thresholds:
        item_str = str(item).strip()
        if "," in item_str:
            for sub in item_str.split(","):
                if sub.strip():
                    results.append(float(sub.strip()))
        elif item_str:
            results.append(float(item_str))

    return sorted(list(set(results))) if results else [0.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Replay-level evaluation of stored predictions: single-region IR and "
            "Intersection@, KBRS cue scores, and multi-region metrics (CWO, M-CTI, "
            "event recall). See the module docstring for the CSV columns."
        )
    )

    # Core identification & source
    p.add_argument("--replays", type=str, nargs="+", default=None, help="Target replay IDs (e.g. 275 1725 3613)")
    p.add_argument("--mode", choices=["feature", "gt", "model"], default="model", help="Evaluation mode (default: model)")
    p.add_argument("--task", choices=["single", "multi", "all"], default="all", help="Task: single, multi, or all (default: all)")
    p.add_argument("--model-name", type=str, default=None, help="Model checkpoint folder name")
    p.add_argument("--epoch", type=int, default=None, help="Model epoch number")

    # Flexible score thresholds (supports both space and comma separation)
    p.add_argument(
        "--score-thresholds",
        "--score-threshold",
        "--thresholds",
        dest="score_thresholds",
        nargs="*",
        default=None,
        help="One or more score thresholds (e.g. 0.0 0.1 0.2 or 0.0,0.1,0.2).",
    )

    # Data paths
    p.add_argument("--input-root", default="/workspace/data/input/dst")
    p.add_argument("--label-root", default="/workspace/data/label/dst")
    p.add_argument("--label-method", default="all_correct")
    p.add_argument("--pred-root", default="/workspace/predictions")
    p.add_argument("--csv-out", default=None, help="Output CSV path")

    # KBRS options
    p.add_argument("--window", default="20,12", help="KBRS window (w,h)")
    p.add_argument("--stride", default="1,1", help="KBRS stride (x,y)")
    p.add_argument("--threshold", type=float, default=0.0, help="KBRS threshold")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--use-file-name", action="store_true")
    p.add_argument("--feature-ext", default=".npy")
    p.add_argument("--skip-kbrs", action="store_true", help="Skip KBRS calculations")
    p.add_argument("--num-workers", type=int, default=0)

    # Multi-Region options
    p.add_argument("--multi-topk", type=int, default=3, help="Top-K predicted viewports to evaluate for Multi-Region metrics (default: 3, set 0 for all)")

    # IC metric kernel options
    # Reproduce earlier evaluator behavior, to re-derive numbers reported from it
    # and measure each fix on its own. Both default to the corrected behavior.
    p.add_argument(
        "--skip-missing-preds",
        action="store_true",
        help="Drop frames with no prediction instead of scoring them 0 (pre-8de59a3/cf46716 behavior)",
    )
    p.add_argument(
        "--legacy-centroid",
        action="store_true",
        help="Take prediction centroids from the stored, edge-clamped box size (pre-0782acd behavior)",
    )
    p.add_argument(
        "--missing-as-corner",
        action="store_true",
        help="Score a frame with no prediction as a viewport at the map corner instead of 0 "
             "(the behavior from cf46716 until this flag was added)",
    )
    p.add_argument(
        "--legacy-anchor",
        action="store_true",
        help="Pass annotation centroids to the kernel evaluator where it expects top-left corners "
             "(the behavior until this flag was added; shifts every window by half a viewport)",
    )
    # IR evaluator geometry; defaults IC_KERNEL_WH, IC_GRID_WH, IC_MAXCOORD_XY
    p.add_argument("--ic-kernel", default=None, help="viewport w,h in tiles (default 20,12)")
    p.add_argument("--ic-grid", default=None, help="tile map w,h (default 128,128)")
    p.add_argument("--ic-maxcoord", default=None, help="largest top-left x,y in map pixels (default 3456,3720)")

    # File-to-file mode (run_kernel_eval)
    p.add_argument("--gt", default=None)
    p.add_argument("--gt-dir", default=None)
    p.add_argument("--pred", nargs="+", default=None)
    p.add_argument("--pred-dir", default=None)
    p.add_argument("--out", default="./results")
    p.add_argument("--name", action="append", default=None)
    p.add_argument("--run-tag", default="")

    args = p.parse_args()
    return args


def run_file_mode(args: argparse.Namespace) -> None:
    """The older file-to-file mode: score prediction files against GT files."""
    th_list = parse_thresholds(args.score_thresholds)
    all_rows, summary_json, csv_path, json_path = run_kernel_eval(
        gt_path=args.gt,
        gt_dir=args.gt_dir,
        pred_files=args.pred,
        pred_dir=args.pred_dir,
        out_dir=args.out,
        names=args.name,
        kernel=args.ic_kernel,
        grid=args.ic_grid,
        maxcoord=args.ic_maxcoord,
        score_thresholds=th_list,
        run_tag=args.run_tag,
    )
    print_section_header("Kernel Metrics Summary by Score Threshold")
    if all_rows:
        cols = list(all_rows[0].keys())
        rb = ReportBlock(title="Kernel IC Metrics", columns=cols, aligns=["left", "right"] + ["right"] * (len(cols) - 2))
        for row in all_rows:
            rb.add_row(*[row.get(c) for c in cols])
        rb.add_note(f"Summary CSV  : {csv_path}")
        rb.add_note(f"Summary JSON : {json_path}")
        rb.print()


def _default_csv_out(args: argparse.Namespace) -> str:
    """/workspace/results/<model-name>_e<epoch>.csv, or <mode>.csv for --mode gt."""
    base_root = "/workspace/results"
    os.makedirs(base_root, exist_ok=True)
    if args.mode == "model" and args.model_name:
        csv_name = f"{args.model_name}_e{args.epoch}.csv"
    else:
        csv_name = f"{args.mode}.csv"
    return os.path.join(base_root, csv_name)


def evaluate_replay(args: argparse.Namespace, replay_id: str, thresholds: List[float]) -> List[Dict]:
    """The CSV rows of one replay, one per score threshold."""
    print(f"\n[Replay] {replay_id}")
    coco_gt = load_coco_gt(
        label_root=args.label_root,
        replay_id=replay_id,
        label_method=args.label_method,
    )
    images = list(coco_gt.dataset.get("images", []))
    if args.max_frames > 0:
        images = images[: args.max_frames]

    preds_by_img: Optional[Dict[int, List[dict]]] = None
    model_tag: Optional[str] = None
    if args.mode == "model":
        preds_by_img = load_coco_preds(
            pred_root=args.pred_root,
            model_name=args.model_name,
            epoch=args.epoch,
            replay_id=replay_id,
            label_method=args.label_method,
        )
        model_tag = f"{args.model_name}_e{args.epoch}"

    # 1. KBRS cue scores, once per replay (task single/all)
    mean_density = mean_centered = mean_mixture = float("nan")
    if args.task in ["single", "all"]:
        if args.skip_kbrs:
            print(f"[Single-Region replay={replay_id}] KBRS skipped (--skip-kbrs)")
        else:
            mean_density, mean_centered, mean_mixture, _ = compute_kbrs_for_replay(
                replay_id=replay_id,
                mode=args.mode,
                coco_gt=coco_gt,
                args=args,
                preds_by_img=preds_by_img,
                model_tag=model_tag,
            )

    # 2. Multi-region metrics, once per replay (task multi/all)
    if args.task in ["multi", "all"]:
        multi_row = compute_multi_region_for_replay(
            replay_id=replay_id,
            coco_gt=coco_gt,
            preds_by_img=preds_by_img,
            multi_topk=getattr(args, "multi_topk", 3),
        )
    else:
        multi_row = dict(NAN_MULTI_ROW)

    # 3. Single-region IR / Intersection@ per score threshold
    rows = []
    for th in thresholds:
        if args.task in ["single", "all"]:
            ic_row = compute_ic_for_replay(
                replay_id=replay_id,
                mode=args.mode,
                coco_gt=coco_gt,
                args=args,
                preds_all=preds_by_img,
                model_tag=model_tag,
                score_thresh=th,
                skip_missing_preds=args.skip_missing_preds,
                legacy_centroid=args.legacy_centroid,
                missing_as_corner=args.missing_as_corner,
                legacy_anchor=args.legacy_anchor,
            )
        else:
            ic_row = {"kernel": "20x12", "score_thresh": float(th), "num_images": len(images)}

        row = dict(ic_row)
        row.update(multi_row)
        row.update({
            "replay": replay_id,
            "score_thresh": float(th),
            "kernel": ic_row.get("kernel"),
            "num_images": ic_row.get("num_images", len(images)),
            "mean_density": mean_density,
            "mean_centeredness": mean_centered,
            "mean_mixture": mean_mixture,
            "mode": args.mode,
            "task": args.task,
            "model_name": args.model_name if args.mode == "model" else None,
            "epoch": args.epoch if args.mode == "model" else None,
        })
        rows.append(row)
    return rows


def write_summary(args: argparse.Namespace, rows: List[Dict]) -> None:
    """Write the rows to --csv-out and print them as a table."""
    df_all = pd.DataFrame(rows).sort_values(by=["replay", "score_thresh"])
    out_dir = os.path.dirname(args.csv_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_all.to_csv(args.csv_out, index=False)

    print_section_header("Replay-Level Evaluation Summary")
    cols = list(df_all.columns)
    rb = ReportBlock(
        title=f"Replay Summary ({args.mode}, task={args.task})",
        columns=cols,
        aligns=["left"] + ["right"] * (len(cols) - 1),
    )
    for _, r in df_all.iterrows():
        rb.add_row(*[r.get(c) for c in cols])
    rb.add_note(f"Saved CSV : {args.csv_out}")
    rb.print()


def main():
    args = parse_args()

    if args.gt or args.gt_dir or (args.pred and not args.replays):
        run_file_mode(args)
        return

    if not args.replays:
        raise ValueError("Please specify replay IDs with --replays (e.g. --replays 275 1725 3613).")
    if args.mode == "model" and (not args.model_name or args.epoch is None):
        raise ValueError("mode=model requires both --model-name and --epoch.")

    thresholds = parse_thresholds(args.score_thresholds)
    print(f"[INFO] Score thresholds to evaluate: {thresholds}")
    if not args.csv_out:
        args.csv_out = _default_csv_out(args)
    print(f"[INFO] Output CSV (replay-level rows): {args.csv_out}")

    rows: List[Dict] = []
    for replay_id in args.replays:
        rows.extend(evaluate_replay(args, str(replay_id), thresholds))

    if not rows:
        print("[Info] No replay rows to write CSV.")
        return
    write_summary(args, rows)


if __name__ == "__main__":
    main()
