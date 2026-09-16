# src/evaluate.py
"""
Unified Evaluation Framework for StarCraft Vision:
Integrates:
  1. Single-Region Finding: KBRS Scores (Density, Centeredness, Mixture) & COCO IC Metrics (Coverage, IR, Multi-Coverage)
  2. Score Threshold Sweep: Detailed IC metrics across multiple confidence thresholds
  3. Multi-Region Finding: CWO (Consensus Overlap), M-CTI (Camera Thrashing/Jerk), Event Recall, Pairwise Overlap
  4. End-to-End Model Benchmark: Direct PyTorch inference and Proposed Video DETR comparison
"""
from __future__ import annotations

import os
import sys
import csv
import json
import time
import argparse
from dataclasses import asdict
from typing import Any, Dict, List, Sequence, Tuple, Optional
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

# Local module imports
from metrics import eval_intersection_run, ImageIR
from metrics.evaluator import MultiRegionEvaluator
from utils.logger import Logger
from utils.report import ReportBlock, print_section_header

# Ensure unbuffered stdout in container/redirection environments
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


# =====================================================================
# 1. COCO & Geometry Helpers
# =====================================================================

def _centroid_from_coco_ann(ann: dict, img_w: int, img_h: int) -> Tuple[float, float]:
    """
    Return centroid (cx, cy) in image pixel coordinates (0..img_w-1, 0..img_h-1)
    Handles 'segmentation' (polygon or RLE) and 'bbox'.
    """
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
            return float(x + w / 2.0), float(y + h / 2.0)

    if "bbox" in ann and ann["bbox"]:
        x, y, w, h = ann["bbox"]
        return float(x + w / 2.0), float(y + h / 2.0)

    return float(img_w) / 2.0, float(img_h) / 2.0


centroid_from_coco_ann = _centroid_from_coco_ann


def coco_to_kernel_labels(
    coco_gt: COCO,
    preds_list: List[dict],
    *,
    x_len: int,
    y_len: int,
    grid_w: int,
    grid_h: int,
    max_x: float,
    max_y: float,
    score_thresh: float = 0.0,
    skip_missing_preds: bool = False,
    return_stats: bool = False,
):
    """
    Convert COCO-style GT + preds into agent-trace tests for kernel-based evaluator.
    Only predictions with score >= score_thresh are considered valid.
    """
    preds_by_img: Dict[int, List[dict]] = {}
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

    for img in coco_gt.dataset.get("images", []):
        image_id = int(img["id"])
        stats["total_frames"] += 1
        img_w = int(img.get("width", grid_w))
        img_h = int(img.get("height", grid_h))

        ann_ids = coco_gt.getAnnIds(imgIds=image_id)
        anns = coco_gt.loadAnns(ann_ids) if ann_ids else []

        img_preds = preds_by_img.get(image_id, [])
        valid_preds = [p for p in img_preds if float(p.get("score", 1.0)) >= score_thresh]

        if len(valid_preds) == 0:
            stats["missing_preds"] += 1
            if skip_missing_preds:
                continue
            dummy_vx, dummy_vy = -9999.0, -9999.0
            agent0 = [{"vpx": dummy_vx, "vpy": dummy_vy}]
        else:
            best_pred = max(valid_preds, key=lambda q: float(q.get("score", 0.0)))
            pcx, pcy = _centroid_from_coco_ann(best_pred, img_w, img_h)
            vx = float(pcx) / max(1, (img_w - x_len)) * max_x
            vy = float(pcy) / max(1, (img_h - y_len)) * max_y
            agent0 = [{"vpx": vx, "vpy": vy}]

        ref_agents: List[List[Dict[str, float]]] = []
        for ann in anns:
            gcx, gcy = _centroid_from_coco_ann(ann, img_w, img_h)
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
    preds_list: List[dict],
    *,
    name: str = "run",
    kernel: Tuple[int, int] = (20, 12),
    grid: Tuple[int, int] = (128, 128),
    maxcoord: Tuple[float, float] = (3456.0, 3720.0),
    skip_missing_preds: bool = False,
    score_thresh: float = 0.0,
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
        return_stats=True,
    )

    if len(labels_tests) == 0:
        empty_row = {
            "name": name,
            "score_thresh": float(score_thresh),
            "kernel": f"{x_len}x{y_len}",
            "num_images": 0,
            "ic@000": float("nan"),
            "ic@030": float("nan"),
            "ic@050": float("nan"),
            "ic_multi": float("nan"),
            "ic_ratio": float("nan"),
            "median_ir": float("nan"),
            "p90_ir": float("nan"),
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
# 2. KBRS metric calculation (density / centeredness / mixture)
# =====================================================================

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
    window_val = getattr(args, "window", "20,12")
    if isinstance(window_val, str):
        win_w, win_h = map(int, window_val.split(","))
    else:
        win_w, win_h = map(int, window_val)

    stride_val = getattr(args, "stride", "1,1")
    if isinstance(stride_val, str):
        stride_x, stride_y = map(int, stride_val.split(","))
    else:
        stride_x, stride_y = map(int, stride_val)

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
                _centroid_from_coco_ann(det, img_w, img_h)
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
) -> Dict[int, List[dict]]:
    pred_dir = os.path.join(
        pred_root, model_name, f"model_{epoch:03d}", f"{replay_id}.rep"
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
    preds_all: Optional[List[dict]] = None,
    model_tag: Optional[str] = None,
    skip_missing_preds: bool = False,
    score_thresh: float = 0.0,
) -> Dict[str, float]:
    """
    Computes IC metrics for a replay sequence with a specified score threshold.
    """
    ic_kernel_val = getattr(args, "ic_kernel", "20,12")
    if isinstance(ic_kernel_val, (list, tuple)):
        ic_x_len, ic_y_len = int(ic_kernel_val[0]), int(ic_kernel_val[1])
    elif isinstance(ic_kernel_val, str) and "," in ic_kernel_val:
        ic_x_len, ic_y_len = map(int, ic_kernel_val.split(","))
    else:
        ic_x_len, ic_y_len = 20, 12

    ic_grid_val = getattr(args, "ic_grid", "128,128")
    if isinstance(ic_grid_val, (list, tuple)):
        ic_grid_w, ic_grid_h = int(ic_grid_val[0]), int(ic_grid_val[1])
    elif isinstance(ic_grid_val, str) and "," in ic_grid_val:
        ic_grid_w, ic_grid_h = map(int, ic_grid_val.split(","))
    else:
        ic_grid_w, ic_grid_h = 128, 128

    ic_max_val = getattr(args, "ic_maxcoord", "3456,3720")
    if isinstance(ic_max_val, (list, tuple)):
        ic_max_x, ic_max_y = float(ic_max_val[0]), float(ic_max_val[1])
    elif isinstance(ic_max_val, str) and "," in ic_max_val:
        ic_max_x, ic_max_y = map(float, ic_max_val.split(","))
    else:
        ic_max_x, ic_max_y = 3456.0, 3720.0

    images = list(coco_gt.dataset.get("images", []))
    max_frames = getattr(args, "max_frames", -1)
    if max_frames > 0:
        images = images[:max_frames]

    num_images = len(images)
    num_preds = len(preds_all) if preds_all else 0

    print(
        f"[IC replay={replay_id}] start "
        f"(mode={mode}, thresh={score_thresh}, kernel={ic_x_len}x{ic_y_len}, "
        f"grid={ic_grid_w}x{ic_grid_h}, maxcoord=({ic_max_x},{ic_max_y}), "
        f"num_images={num_images}, num_preds={num_preds}, skip_missing={skip_missing_preds})"
    )

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

        return {
            "kernel": f"{ic_x_len}x{ic_y_len}",
            "score_thresh": float(score_thresh),
            "num_images": num_images,
            "ic@000": float("nan"),
            "ic@030": float("nan"),
            "ic@050": float("nan"),
            "ic_multi": float("nan"),
            "ic_ratio": float("nan"),
            "median_ir": float("nan"),
            "p90_ir": float("nan"),
        }

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
        )
        print(
            f"[IC replay={replay_id}] thresh={score_thresh} done (t={time.time() - t_ic:.2f}s) "
            f"-> IC@0.0={row.get('ic@000', float('nan')):.3f}, IC@0.5={row.get('ic@050', float('nan')):.3f}, IC_multi={row.get('ic_multi', float('nan')):.3f}",
            flush=True,
        )
        return row

    return {
        "kernel": f"{ic_x_len}x{ic_y_len}",
        "score_thresh": float(score_thresh),
        "num_images": num_images,
        "ic@000": float("nan"),
        "ic@030": float("nan"),
        "ic@050": float("nan"),
        "ic_multi": float("nan"),
        "ic_ratio": float("nan"),
        "median_ir": float("nan"),
        "p90_ir": float("nan"),
    }


def compute_multi_region_for_replay(
    replay_id: str,
    coco_gt: COCO,
    preds_by_img: Optional[Dict[int, List[dict]]],
    grid_size: Tuple[int, int] = (128, 128),
) -> Dict[str, float]:
    images = list(coco_gt.dataset.get("images", []))
    if not images:
        return {
            "cwo": float("nan"),
            "m_cti": float("nan"),
            "jerk": float("nan"),
            "jump_rate": float("nan"),
            "event_recall": float("nan"),
            "pairwise_overlap": float("nan"),
        }

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

    print(f"[Multi-Region replay={replay_id}] Preparing sequences ({num_images} frames)...", flush=True)
    t0 = time.time()
    step_interval = max(5000, num_images // 10)

    for idx, img in enumerate(images):
        if (idx + 1) % step_interval == 0 or (idx + 1) == num_images:
            pct = (idx + 1) / num_images * 100.0
            print(f"  [Multi-Region Prep] {idx + 1}/{num_images} frames ({pct:.1f}%)", flush=True)

        img_id = int(img["id"])
        h_img = float(img.get("height", grid_h))
        w_img = float(img.get("width", grid_w))

        img_boxes = []
        if preds_by_img and img_id in preds_by_img:
            for det in preds_by_img[img_id]:
                b = det.get("bbox", None)
                if b is not None and len(b) == 4:
                    x, y, w, h = b
                    x1 = (x / w_img) * grid_w
                    y1 = (y / h_img) * grid_h
                    x2 = ((x + w) / w_img) * grid_w
                    y2 = ((y + h) / h_img) * grid_h
                    img_boxes.append([x1, y1, x2, y2])

        if not img_boxes:
            img_boxes = [[0.0, 0.0, 32.0, 32.0]]
        viewports_seq.append(np.array(img_boxes, dtype=np.float32))

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
        f"-> CWO={metrics.get('cwo', 0.0):.4f}, M-CTI={metrics.get('m_cti', 0.0):.4f}, Recall={metrics.get('event_recall', 0.0):.4f}",
        flush=True,
    )
    return metrics


# =====================================================================
# 4. Standalone File-based Kernel Evaluation (Legacy evaluate.py API)
# =====================================================================

def run_kernel_eval(
    gt_path: Optional[str] = None,
    gt_dir: Optional[str] = None,
    pred_files: Optional[Sequence[str]] = None,
    pred_dir: Optional[str] = None,
    out_dir: str = "./results",
    names: Optional[Sequence[str]] = None,
    kernel: str = "20,12",
    grid: str = "128,128",
    maxcoord: str = "3456,3720",
    score_thresholds: Sequence[float] = (0.0,),
    per_image_csv: bool = False,
    batch_size: int = 0,
    run_tag: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str, str]:
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

    x_len, y_len = map(int, kernel.split(","))
    grid_w, grid_h = map(int, grid.split(","))
    max_x, max_y = map(float, maxcoord.split(","))

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
# 5. E2E Benchmark Runner (Integrated from run_benchmark.py)
# =====================================================================

def run_e2e_benchmark(args: argparse.Namespace):
    """
    Executes End-to-End model benchmark inference & Proposed model comparison.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys_paths = [root_dir, os.path.join(root_dir, "src")]
    for p in sys_paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    from dataset.custom_penn_fudan import CustomPennFudanDataset
    from models import build_model, ProbabilisticVideoDETR

    window_size = getattr(args, "window_size", None)
    if window_size is None:
        model_lower = (args.model_name or "").lower()
        if "win1" in model_lower:
            window_size = 1
        elif "win4" in model_lower:
            window_size = 4
        else:
            window_size = 4

    out_json = getattr(args, "output_json", None)
    if not out_json:
        bench_dir = "/workspace/results/benchmark"
        os.makedirs(bench_dir, exist_ok=True)
        out_json = os.path.join(bench_dir, f"{args.model_name}_e{args.epoch}.json")

    print("=" * 85)
    print(f"🚀 Running Evaluation Benchmark (Task Mode: {args.task.upper()})")
    print(f"[*] Target Model    : {args.model_name} (Epoch {args.epoch})")
    print(f"[*] Window Size     : {window_size}")
    print(f"[*] Output Path     : {out_json}")
    print(f"[*] Target Replays  : {args.replays}")
    print("=" * 85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print("[*] Loading dataset windows...")
    try:
        ds = CustomPennFudanDataset(
            input_root=args.input_root,
            label_root=args.label_root,
            label_method=args.label_method,
            training_ids=args.replays,
            window_size=window_size,
            include_components=getattr(args, "include_components", None),
            interval=1,
            training=False,
            verbose=True,
        )
    except Exception as e:
        local_input = os.path.join(root_dir, "data/input/dst")
        local_label = os.path.join(root_dir, "data/label/dst")
        ds = CustomPennFudanDataset(
            input_root=local_input,
            label_root=local_label,
            label_method=args.label_method,
            training_ids=args.replays,
            window_size=window_size,
            include_components=getattr(args, "include_components", None),
            interval=1,
            training=False,
            verbose=True,
        )

    # Replay metrics computation
    replay_results = []
    base_avg: Dict[str, float] = {}

    for replay_id in args.replays:
        replay_id = str(replay_id)
        coco_gt = load_coco_gt(args.label_root, replay_id, args.label_method)
        preds_by_img = load_coco_preds(args.pred_root, args.model_name, args.epoch, replay_id, args.label_method)

        preds_all = []
        for dets in preds_by_img.values():
            preds_all.extend(dets)

        ic_row = compute_ic_for_replay(
            replay_id=replay_id,
            mode="model",
            coco_gt=coco_gt,
            args=args,
            preds_all=preds_all if preds_all else None,
            model_tag=f"{args.model_name}_e{args.epoch}",
        )

        multi_row = compute_multi_region_for_replay(
            replay_id=replay_id,
            coco_gt=coco_gt,
            preds_by_img=preds_by_img,
        )

        combined = dict(ic_row)
        combined.update(multi_row)
        combined["replay"] = replay_id
        replay_results.append(combined)

    df_res = pd.DataFrame(replay_results)
    for col in df_res.select_dtypes(include=[np.number]).columns:
        base_avg[col] = float(df_res[col].dropna().mean()) if not df_res[col].dropna().empty else float("nan")

    # Display comparison if requested
    print_section_header(f"BENCHMARK RESULTS SUMMARY: {args.model_name} (Epoch {args.epoch})")
    cols = list(df_res.columns)
    rb = ReportBlock(
        title=f"Benchmark Summary ({args.model_name})",
        columns=cols,
        aligns=["left"] + ["right"] * (len(cols) - 1),
    )
    for _, r in df_res.iterrows():
        rb.add_row(*[r.get(c) for c in cols])
    rb.print()

    # Save outputs
    out_dir = os.path.dirname(out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"replays": replay_results, "summary_averages": base_avg}, f, indent=2)
    print(f"[*] Saved benchmark summary JSON: {out_json}")


# =====================================================================
# 6. Flexible CLI Parser & Main Entry Point
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
            "Unified Evaluator for Single-Region Finding (KBRS, COCO IC), "
            "Multi-Region Finding (CWO, M-CTI, Event Recall), & End-to-End Model Benchmarking."
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

    # IC metric kernel options
    p.add_argument("--ic-kernel", default="20,12")
    p.add_argument("--ic-grid", default="128,128")
    p.add_argument("--ic-maxcoord", default="3456,3720")

    # Benchmark & Comparison options
    p.add_argument("--benchmark", action="store_true", help="Run in End-to-End benchmark mode")
    p.add_argument("--compare-proposed", action="store_true", help="Compare model against Proposed Video DETR+CVAE")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--include-components", type=str, nargs="+", default=None)
    p.add_argument("--output-json", type=str, default=None)

    # Legacy standalone evaluate.py options for compatibility
    p.add_argument("--gt", default=None)
    p.add_argument("--gt-dir", default=None)
    p.add_argument("--pred", nargs="+", default=None)
    p.add_argument("--pred-dir", default=None)
    p.add_argument("--out", default="./results")
    p.add_argument("--name", action="append", default=None)
    p.add_argument("--run-tag", default="")

    args = p.parse_args()
    return args


def main():
    args = parse_args()

    # Legacy file-to-file evaluate mode
    if args.gt or args.gt_dir or (args.pred and not args.replays):
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
        return

    # Benchmark E2E mode
    if args.benchmark or args.compare_proposed:
        if not args.replays:
            args.replays = ["1725"]
        run_e2e_benchmark(args)
        return

    # Standard Unified Replay-level Evaluation
    if not args.replays:
        raise ValueError("Please specify replay IDs with --replays (e.g. --replays 275 1725 3613).")

    if args.mode == "model" and (not args.model_name or args.epoch is None):
        raise ValueError("mode=model requires both --model-name and --epoch.")

    thresholds = parse_thresholds(args.score_thresholds)
    print(f"[INFO] Score thresholds to evaluate: {thresholds}")

    # Set default CSV output path
    if not args.csv_out:
        base_root = "/workspace/results"
        os.makedirs(base_root, exist_ok=True)
        if args.mode == "model" and args.model_name:
            csv_name = f"{args.model_name}_e{args.epoch}.csv"
        else:
            csv_name = f"{args.mode}.csv"
        args.csv_out = os.path.join(base_root, csv_name)

    print(f"[INFO] Output CSV (replay-level rows): {args.csv_out}")

    replay_rows: List[Dict] = []

    for replay_id in args.replays:
        replay_id = str(replay_id)
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
        preds_all: List[dict] = []
        model_tag: Optional[str] = None

        if args.mode == "model":
            preds_by_img = load_coco_preds(
                pred_root=args.pred_root,
                model_name=args.model_name,
                epoch=args.epoch,
                replay_id=replay_id,
                label_method=args.label_method,
            )
            for dets in preds_by_img.values():
                preds_all.extend(dets)
            model_tag = f"{args.model_name}_e{args.epoch}"

        # 1. KBRS metric (computed once per replay if task in single/all)
        if args.task in ["single", "all"]:
            if args.skip_kbrs:
                mean_density = float("nan")
                mean_centered = float("nan")
                mean_mixture = float("nan")
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
        else:
            mean_density = float("nan")
            mean_centered = float("nan")
            mean_mixture = float("nan")

        # 2. Multi-Region metric (computed once per replay if task in multi/all)
        if args.task in ["multi", "all"]:
            multi_row = compute_multi_region_for_replay(
                replay_id=replay_id,
                coco_gt=coco_gt,
                preds_by_img=preds_by_img,
            )
        else:
            multi_row = {
                "cwo": float("nan"),
                "m_cti": float("nan"),
                "jerk": float("nan"),
                "jump_rate": float("nan"),
                "event_recall": float("nan"),
                "pairwise_overlap": float("nan"),
            }

        # 3. Single-Region IC metrics per threshold
        for th in thresholds:
            if args.task in ["single", "all"]:
                ic_row = compute_ic_for_replay(
                    replay_id=replay_id,
                    mode=args.mode,
                    coco_gt=coco_gt,
                    args=args,
                    preds_all=preds_all if preds_all else None,
                    model_tag=model_tag,
                    score_thresh=th,
                )
            else:
                ic_row = {"kernel": "20x12", "score_thresh": float(th), "num_images": len(images)}

            # Combine row
            final_row = dict(ic_row)
            final_row.update(multi_row)
            final_row.update({
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
            replay_rows.append(final_row)

    if not replay_rows:
        print("[Info] No replay rows to write CSV.")
        return

    df_all = pd.DataFrame(replay_rows).sort_values(by=["replay", "score_thresh"])
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


if __name__ == "__main__":
    main()
