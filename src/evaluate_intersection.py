#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_intersection.py
------------------------
Custom evaluator for region overlap metrics used in Observer-style tasks.

Core outputs per run (always):
- ic@000 : fraction IR>0
- ic@030 : fraction IR>=0.30
- ic@050 : fraction IR>=0.50
- ic_multi : fraction of images where >=2 predicted regions intersect GT (>0 each)
- ic_ratio : mean IR

Kernel metrics (new):
- density     : |T ∩ W| / |W|
- centeredness: 1 - (dist(centroid(T∩W), center(W)) / (0.5*diag(W)))
- mixture     : density * centeredness
  where W is a rectangular window of size (W,H) centered on a reference point,
  and T is chosen via --window-target:
    * gt        : union(GT)                  (default)
    * intersect : intersection(P, GT)        (requires prediction)

Batch means:
- If --batch-size > 0, compute means of [density, centeredness, mixture] per
  sequential chunk of that many evaluated images and write {out}/batch_metrics.csv

Usage
=====
python evaluate_intersection.py \
  --gt path/to/instances.json \
  --pred path/to/run1.json --name run1 \
  --out ./eval_ir \
  --denom gt --pred-agg best \
  --kernel 20,12 --window-source pred --window-target gt \
  --batch-size 64 --per-image

Notes
=====
- GT and predictions can be bbox or segmentation; everything is converted to RLE.
- Window is clipped to image borders.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from pycocotools.coco import COCO
    import pycocotools.mask as mask_util
except Exception as e:
    print("ERROR: pycocotools is required. pip install pycocotools", file=sys.stderr)
    raise

# -------------------- Utilities --------------------

def _poly_to_rle(poly, h, w):
    if isinstance(poly, dict) and 'counts' in poly:
        rle = poly
        if isinstance(rle['counts'], list):
            rle = mask_util.frPyObjects(rle, h, w)
        return rle
    if isinstance(poly, list):
        rles = mask_util.frPyObjects(poly, h, w)
        rle = mask_util.merge(rles)
        return rle
    raise ValueError(f"Unsupported segmentation format: {type(poly)}")

def _anno_to_rle(anno, h, w):
    if 'segmentation' in anno and anno['segmentation'] is not None:
        seg = anno['segmentation']
        if isinstance(seg, list):
            rle = _poly_to_rle(seg, h, w)
        elif isinstance(seg, dict):
            rle = seg
            if isinstance(seg.get('counts', None), list):
                rle = mask_util.frPyObjects(rle, h, w)
        else:
            raise ValueError("Unknown segmentation type in annotation.")
        return rle
    elif 'bbox' in anno and anno['bbox'] is not None:
        x, y, wbox, hbox = anno['bbox']
        x0 = max(0, int(np.floor(x)))
        y0 = max(0, int(np.floor(y)))
        x1 = min(w, int(np.ceil(x + wbox)))
        y1 = min(h, int(np.ceil(y + hbox)))
        if x1 <= x0 or y1 <= y0:
            return mask_util.encode(np.zeros((h, w), dtype=np.uint8, order='F'))
        m = np.zeros((h, w), dtype=np.uint8)
        m[y0:y1, x0:x1] = 1
        rle = mask_util.encode(np.asfortranarray(m))
        return rle
    else:
        raise ValueError("Annotation lacks both 'segmentation' and 'bbox'.")

def _union_rles(rles: List[dict], h: int, w: int) -> dict:
    if not rles:
        return mask_util.encode(np.zeros((h, w), dtype=np.uint8, order='F'))
    return mask_util.merge(rles)

def _area(rle: dict) -> float:
    return float(mask_util.area(rle))

def _intersect(rle1: dict, rle2: dict) -> dict:
    return mask_util.merge([rle1, rle2], intersect=True)

def _bbox_from_rle(rle: dict) -> Tuple[int,int,int,int]:
    bb = mask_util.toBbox(rle)  # (x,y,w,h) float
    x, y, w, h = bb
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = int(np.ceil(x + w))
    y1 = int(np.ceil(y + h))
    return x0, y0, x1, y1

def _centroid_from_rle(rle: dict) -> Tuple[float, float]:
    # approximate centroid from bbox midpoint for speed; if area small, it still works
    x0, y0, x1, y1 = _bbox_from_rle(rle)
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0

def _window_mask(center_x: float, center_y: float, win_w: int, win_h: int, H: int, W: int):
    # Build binary mask for window W centered at (cx,cy), clipped to image
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
        return mask_util.encode(np.asfortranarray(m)), 0.0, (x0,y0,x1,y1)
    m = np.zeros((H, W), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return mask_util.encode(np.asfortranarray(m)), float((x1-x0)*(y1-y0)), (x0,y0,x1,y1)

def _window_center(window_source: str, P: Optional[dict], G: dict) -> Tuple[float, float]:
    if window_source == 'pred' and P is not None:
        return _centroid_from_rle(P)
    # fallback: GT center
    return _centroid_from_rle(G)

# -------------------- Metrics --------------------

def intersection_ratio(P: dict, G: dict, denom: str) -> float:
    inter = _intersect(P, G)
    inter_area = _area(inter)
    if denom == 'gt':
        D = _area(G)
    elif denom == 'pred':
        D = _area(P)
    elif denom == 'union':
        D = _area(P) + _area(G) - inter_area
    else:
        raise ValueError("Unknown denom, choose from {'gt','pred','union'}")
    if D <= 0:
        return 0.0
    return float(inter_area / D)

def kernel_scores(center_x: float, center_y: float, win_w: int, win_h: int,
                  H: int, W: int, T_mask: dict) -> Tuple[float, float, float]:
    """
    Compute density/centeredness/mixture inside window W for target mask T.
    """
    Wmask, Warea, (x0,y0,x1,y1) = _window_mask(center_x, center_y, win_w, win_h, H, W)
    if Warea <= 0:
        return 0.0, 0.0, 0.0
    TinW = _intersect(T_mask, Wmask)
    t_area = _area(TinW)
    density = float(t_area / Warea)

    # centeredness via centroid distance normalization by half-diagonal
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    if t_area <= 0:
        centeredness = 0.0
    else:
        # centroid approximated via bbox midpoint of T∩W
        tcx, tcy = _centroid_from_rle(TinW)
        dist = float(np.hypot(tcx - cx, tcy - cy))
        half_diag = float(np.hypot((x1-x0), (y1-y0)) / 2.0)
        centeredness = max(0.0, 1.0 - (dist / half_diag)) if half_diag > 0 else 0.0

    mixture = density * centeredness
    return density, centeredness, mixture

# -------------------- Core Evaluation --------------------

@dataclass
class ImageIR:
    image_id: int
    width: int
    height: int
    ir: float
    overlap_count: int
    density: float
    centeredness: float
    mixture: float

def eval_run(
    coco_gt: COCO,
    preds: List[dict],
    denom: str = 'gt',
    pred_agg: str = 'best',
    cat_id: Optional[int] = None,
    window_source: str = 'pred',
    window_target: str = 'gt',
    win_w: int = 20,
    win_h: int = 12,
) -> Tuple[List[ImageIR], Dict[str, float]]:
    gt_by_img: Dict[int, List[dict]] = {}
    for ann in coco_gt.dataset['annotations']:
        if cat_id is not None and ann.get('category_id') != cat_id:
            continue
        gt_by_img.setdefault(ann['image_id'], []).append(ann)

    img_meta: Dict[int, Tuple[int, int]] = {}
    for img in coco_gt.dataset['images']:
        img_meta[img['id']] = (img['height'], img['width'])

    pred_by_img: Dict[int, List[dict]] = {}
    for d in preds:
        if cat_id is not None and d.get('category_id') != cat_id:
            continue
        pred_by_img.setdefault(d['image_id'], []).append(d)

    per_image: List[ImageIR] = []
    for image_id, (H, W) in img_meta.items():
        gt_list = gt_by_img.get(image_id, [])
        if len(gt_list) == 0:
            continue

        G = _union_rles([_anno_to_rle(ann, H, W) for ann in gt_list], H, W)
        preds_list = pred_by_img.get(image_id, [])

        oc = 0
        ir_val = 0.0
        P_sel = None

        if len(preds_list) > 0:
            for p in preds_list:
                Pp = _anno_to_rle(p, H, W)
                if _area(_intersect(Pp, G)) > 0:
                    oc += 1

            if pred_agg == 'union':
                P_union = _union_rles([_anno_to_rle(p, H, W) for p in preds_list], H, W)
                ir_val = intersection_ratio(P_union, G, denom)
                P_sel = P_union
            elif pred_agg == 'best':
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

        # Choose window center and target mask for kernel scores
        cx, cy = _window_center(window_source, P_sel, G)
        if window_target == 'gt':
            Tmask = G
        elif window_target == 'intersect' and P_sel is not None:
            Tmask = _intersect(P_sel, G)
        else:
            Tmask = G  # fallback

        dens, cent, mix = kernel_scores(cx, cy, win_w, win_h, H, W, Tmask)

        per_image.append(ImageIR(
            image_id=image_id, width=W, height=H,
            ir=ir_val, overlap_count=oc,
            density=dens, centeredness=cent, mixture=mix
        ))

    if len(per_image) == 0:
        raise RuntimeError("No images with GT found. Check category filter or inputs.")

    # Aggregates
    ir_values = np.array([x.ir for x in per_image], dtype=float)
    mean_ir = float(np.mean(ir_values))
    median_ir = float(np.median(ir_values))
    p90_ir = float(np.percentile(ir_values, 90))
    coverage_any = float(np.mean(ir_values > 0.0))
    multi_cov = float(np.mean(np.array([x.overlap_count for x in per_image]) >= 2))

    dens_vals = np.array([x.density for x in per_image], dtype=float)
    cent_vals = np.array([x.centeredness for x in per_image], dtype=float)
    mix_vals  = np.array([x.mixture for x in per_image], dtype=float)

    aggregates = {
        'mean_ir': mean_ir,
        'median_ir': median_ir,
        'p90_ir': p90_ir,
        'coverage_any': coverage_any,
        'multi_coverage': multi_cov,
        'num_images': int(len(per_image)),
        'mean_density': float(np.mean(dens_vals)),
        'mean_centeredness': float(np.mean(cent_vals)),
        'mean_mixture': float(np.mean(mix_vals)),
    }
    return per_image, aggregates

def pretty_print_table(rows: List[dict], headers: List[str]) -> None:
    col_widths = {h: max(len(h), *(len(f"{r.get(h,'')}") for r in rows)) for h in headers}
    sep = " | "
    line = "-+-".join("-" * col_widths[h] for h in headers)

    def fmt_row(r: dict) -> str:
        return sep.join(f"{str(r.get(h, '')):<{col_widths[h]}}" for h in headers)

    print(sep.join(f"{h:<{col_widths[h]}}" for h in headers))
    print(line)
    for r in rows:
        print(fmt_row(r))

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate Intersection Ratio and kernel metrics with custom IC outputs.")
    ap.add_argument("--gt", required=True, help="Path to COCO instances json (GT).")
    ap.add_argument("--pred", action="append", required=True, help="Path to prediction json (COCO results). Can repeat.")
    ap.add_argument("--name", action="append", help="Name for each prediction; defaults to basename.")
    ap.add_argument("--out", default="./eval_ir_out", help="Output directory.")
    ap.add_argument("--denom", choices=["gt", "pred", "union"], default="gt", help="Denominator for IR.")
    ap.add_argument("--pred-agg", choices=["best", "union"], default="best", help="Aggregate predictions per image.")
    ap.add_argument("--cat-id", type=int, default=None, help="Filter a specific category id (optional).")
    ap.add_argument("--per-image", action="store_true", help="Write per-image CSV as well.")
    ap.add_argument("--kernel", default="20,12", help="Window size W,H (pixels).")
    ap.add_argument("--window-source", choices=["pred","gt"], default="pred", help="Center the window on which mask.")
    ap.add_argument("--window-target", choices=["gt","intersect"], default="gt", help="Target mask for kernel metrics.")
    ap.add_argument("--batch-size", type=int, default=0, help="If >0, compute per-batch means for kernel metrics.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    coco_gt = COCO(args.gt)

    pred_paths = args.pred
    names = args.name or []
    while len(names) < len(pred_paths):
        names.append(os.path.splitext(os.path.basename(pred_paths[len(names)]))[0])

    try:
        win_w, win_h = [int(x.strip()) for x in args.kernel.split(",")]
    except Exception:
        raise ValueError("--kernel must be 'W,H' with integers, e.g., 20,12")

    all_rows = []
    summary_json = {}
    per_image_records = {}

    for pred_path, name in zip(pred_paths, names):
        with open(pred_path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        if isinstance(preds, dict):
            raise ValueError("Prediction file must be a list of COCO-style detections (not a dict).")

        per_image, agg = eval_run(
            coco_gt=coco_gt,
            preds=preds,
            denom=args.denom,
            pred_agg=args.pred_agg,
            cat_id=args.cat_id,
            window_source=args.window_source,
            window_target=args.window_target,
            win_w=win_w,
            win_h=win_h,
        )

        ir_values = np.array([x.ir for x in per_image], dtype=float)

        ic000 = float(np.mean(ir_values > 0.0))
        ic030 = float(np.mean(ir_values >= 0.30))
        ic050 = float(np.mean(ir_values >= 0.50))
        ic_multi = float(agg['multi_coverage'])
        ic_ratio = float(agg['mean_ir'])

        row = {
            "name": name,
            "denom": args.denom,
            "pred_agg": args.pred_agg,
            "kernel": f"{win_w}x{win_h}",
            "window_source": args.window_source,
            "window_target": args.window_target,
            "num_images": agg["num_images"],
            "ic@000": ic000,
            "ic@030": ic030,
            "ic@050": ic050,
            "ic_multi": ic_multi,
            "ic_ratio": ic_ratio,
            "mean_density": agg["mean_density"],
            "mean_centeredness": agg["mean_centeredness"],
            "mean_mixture": agg["mean_mixture"],
            "median_ir": agg["median_ir"],
            "p90_ir": agg["p90_ir"],
        }
        all_rows.append(row)
        summary_json[name] = row

        # store per-image
        per_image_records[name] = per_image

        # per-image CSV
        if args.per_image:
            import csv
            per_csv = os.path.join(args.out, f"{name}_per_image.csv")
            with open(per_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["image_id", "width", "height", "ir", "overlap_count", "density", "centeredness", "mixture"])
                w.writeheader()
                for it in per_image:
                    w.writerow(asdict(it))

        # per-batch metrics
        if args.batch_size and args.batch_size > 0:
            import csv
            dens_vals = np.array([x.density for x in per_image], dtype=float)
            cent_vals = np.array([x.centeredness for x in per_image], dtype=float)
            mix_vals  = np.array([x.mixture for x in per_image], dtype=float)

            n = len(per_image)
            bs = int(args.batch_size)
            rows = []
            for i in range(0, n, bs):
                j = min(i + bs, n)
                rows.append({
                    "name": name,
                    "batch_index": i // bs,
                    "start_idx": i,
                    "end_idx": j - 1,
                    "batch_size": j - i,
                    "mean_density": float(np.mean(dens_vals[i:j])),
                    "mean_centeredness": float(np.mean(cent_vals[i:j])),
                    "mean_mixture": float(np.mean(mix_vals[i:j])),
                })
            batch_csv = os.path.join(args.out, f"{name}_batch_metrics.csv")
            with open(batch_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for r in rows:
                    w.writerow(r)

    # Save summary CSV/JSON
    import csv
    csv_path = os.path.join(args.out, "summary.csv")
    hdr = list(all_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    json_path = os.path.join(args.out, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print("\n=== Intersection & Kernel Metrics Summary ===")
    pretty_print_table(all_rows, headers=hdr)
    print(f"\nSaved summary CSV → {csv_path}")
    print(f"Saved summary JSON → {json_path}")

if __name__ == "__main__":
    main()
