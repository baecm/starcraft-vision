#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_intersection_standalone.py
===================================
Universal entry-point to compute Intersection metrics.
Supports two modes:
  (A) GT + Prediction JSON(s)  --> classic evaluation (same as evaluate_intersection.py)
  (B) GT + Model + Dataset     --> run inference internally, then evaluate

Mode A (predictions):
---------------------
python evaluate_intersection_standalone.py \
  --gt path/to/instances.json \
  --pred path/to/run1.json --name run1 \
  --out ./eval_ir

Mode B (model+dataset):
-----------------------
# Provide a Python file that implements build_model() and build_val_loader()
python evaluate_intersection_standalone.py \
  --gt path/to/instances.json \
  --exp path/to/your_experiment.py \
  --checkpoint path/to/weights.pth \
  --out ./eval_ir --score-thresh 0.05 --max-dets 300 \
  --denom gt --pred-agg best \
  --kernel 20,12 --window-source pred --window-target gt \
  --batch-size 64 --per-image

Experiment module API (minimal):
--------------------------------
def build_model(checkpoint: Optional[str] = None): -> torch.nn.Module
def build_val_loader() -> torch.utils.data.DataLoader
def get_device() -> torch.device   # optional (default: cuda if available else cpu)

Notes
-----
- Outputs 'summary.csv/json', and optional per-image/batch CSVs, same as evaluate_intersection.py.
- Depends on: torch, pycocotools, numpy.
"""

from __future__ import annotations
import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Local imports (in same folder): eval_api + evaluate_intersection (for CLI fallback)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from eval_api import collect_predictions, evaluate_intersection_metrics

def _import_from_path(path: str):
    spec = importlib.util.spec_from_file_location("user_experiment", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def _device_from_exp(exp_mod):
    # optional get_device()
    if hasattr(exp_mod, "get_device"):
        return exp_mod.get_device()
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_mode_predictions(args):
    # Back-compat CLI using evaluate_intersection.py logic by importing its main,
    # but we reimplement the minimal summarization to keep things local.
    # We'll compute metrics for each pred and save CSV/JSON.
    from pycocotools.coco import COCO
    import evaluate_intersection as EV  # must be colocated
    
    os.makedirs(args.out, exist_ok=True)
    coco_gt = COCO(args.gt)

    pred_paths = args.pred
    names = args.name or []
    while len(names) < len(pred_paths):
        names.append(os.path.splitext(os.path.basename(pred_paths[len(names)]))[0])

    all_rows = []
    summary_json = {}
    for pred_path, name in zip(pred_paths, names):
        with open(pred_path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        if isinstance(preds, dict):
            raise ValueError("Prediction file must be a list of COCO-style detections (not a dict).")

        per_image, agg = EV.eval_run(
            coco_gt=coco_gt,
            preds=preds,
            denom=args.denom,
            pred_agg=args.pred_agg,
            cat_id=args.cat_id,
            window_source=args.window_source,
            window_target=args.window_target,
            win_w=args.kernel[0],
            win_h=args.kernel[1],
        )

        ir_values = np.array([x.ir for x in per_image], dtype=float)
        ic000 = float(np.mean(ir_values > 0.0))
        ic030 = float(np.mean(ir_values >= 0.30))
        ic050 = float(np.mean(ir_values >= 0.50))

        row = {
            "name": name,
            "denom": args.denom,
            "pred_agg": args.pred_agg,
            "kernel": f"{args.kernel[0]}x{args.kernel[1]}",
            "window_source": args.window_source,
            "window_target": args.window_target,
            "num_images": agg["num_images"],
            "ic@000": ic000,
            "ic@030": ic030,
            "ic@050": ic050,
            "ic_multi": float(agg["multi_coverage"]),
            "ic_ratio": float(agg["mean_ir"]),
            "mean_density": float(agg["mean_density"]),
            "mean_centeredness": float(agg["mean_centeredness"]),
            "mean_mixture": float(agg["mean_mixture"]),
            "median_ir": float(agg["median_ir"]),
            "p90_ir": float(agg["p90_ir"]),
        }
        all_rows.append(row)
        summary_json[name] = row

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

    print(f"\nSaved summary CSV → {csv_path}")
    print(f"Saved summary JSON → {json_path}")

def run_mode_model(args):
    # Dynamic import of user experiment module
    exp = _import_from_path(args.exp)
    import torch

    device = _device_from_exp(exp)
    model = exp.build_model(checkpoint=args.checkpoint).to(device)
    val_loader = exp.build_val_loader()

    # Collect predictions
    preds = collect_predictions(
        model=model,
        dataloader=val_loader,
        device=device,
        score_thresh=args.score_thresh,
        max_dets=args.max_dets,
        with_masks=not args.no_masks,
    )

    # Evaluate
    row = evaluate_intersection_metrics(
        gt_json_path=args.gt,
        predictions=preds,
        module_path=os.path.dirname(__file__),
        denom=args.denom,
        pred_agg=args.pred_agg,
        cat_id=args.cat_id,
        window_source=args.window_source,
        window_target=args.window_target,
        kernel=tuple(args.kernel),
    )

    # Save
    os.makedirs(args.out, exist_ok=True)
    import csv, json as _json
    csv_path = os.path.join(args.out, "summary.csv")
    hdr = list(row.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerow(row)

    json_path = os.path.join(args.out, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump({"model_eval": row}, f, ensure_ascii=False, indent=2)

    print("Evaluation result:", row)
    print(f"Saved → {csv_path} / {json_path}")

def parse_args():
    ap = argparse.ArgumentParser(description="Standalone evaluator for Intersection metrics (predictions or model+dataset).")
    ap.add_argument("--gt", required=True, help="Path to COCO instances json (GT).")
    ap.add_argument("--out", default="./eval_ir_out", help="Output directory.")

    # Mode A (predictions)
    ap.add_argument("--pred", action="append", help="Prediction json (COCO results). Can repeat.")
    ap.add_argument("--name", action="append", help="Name for each prediction (defaults to basename).")

    # Mode B (model+dataset)
    ap.add_argument("--exp", help="Path to experiment module .py implementing build_model/build_val_loader.")
    ap.add_argument("--checkpoint", default=None, help="Optional model checkpoint path.")
    ap.add_argument("--score-thresh", type=float, default=0.0, help="Score threshold for detections.")
    ap.add_argument("--max-dets", type=int, default=None, help="Cap number of detections per image.")
    ap.add_argument("--no-masks", action="store_true", help="Do not export masks (bbox-only).")

    # Common metric options
    ap.add_argument("--denom", choices=["gt", "pred", "union"], default="gt", help="Denominator for IR.")
    ap.add_argument("--pred-agg", choices=["best", "union"], default="best", help="Aggregate predictions per image.")
    ap.add_argument("--cat-id", type=int, default=None, help="Filter category id (optional).")
    ap.add_argument("--per-image", action="store_true", help="(predictions mode) write per-image CSV as well.")
    ap.add_argument("--batch-size", type=int, default=0, help="(predictions mode) per-batch means for kernel metrics.")
    ap.add_argument("--kernel", default="20,12", help="Window size W,H (pixels).")
    ap.add_argument("--window-source", choices=["pred","gt"], default="pred", help="Center the window on which mask.")
    ap.add_argument("--window-target", choices=["gt","intersect"], default="gt", help="Target mask for kernel metrics.")
    args = ap.parse_args()

    # parse kernel
    if isinstance(args.kernel, str):
        try:
            W,H = [int(x.strip()) for x in args.kernel.split(",")]
            args.kernel = (W,H)
        except Exception:
            raise ValueError("--kernel must be 'W,H' with integers, e.g., 20,12")

    return args

def main():
    args = parse_args()

    # Mode dispatch
    if args.pred is not None and len(args.pred) > 0:
        run_mode_predictions(args)
    elif args.exp is not None:
        run_mode_model(args)
    else:
        raise SystemExit("Please provide either --pred (prediction file[s]) or --exp (experiment module).")

if __name__ == "__main__":
    main()
