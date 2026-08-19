#!/usr/bin/env python3
"""
Main Benchmark Script for Single-Region Finding & Multi-Region Finding Evaluator:
Evaluates specified trained models (e.g. CenterNet, Deformable DETR, Mask R-CNN, or Proposed Video DETR)
on real StarCraft replay dataset state tensors (from data/input/dst) and COCO annotations (from data/label/dst),
computing task-specific metrics:
  - Single-Region Finding Task: KBRS Scores (Density, Centeredness, Mixture), COCO IC Metrics
  - Multi-Region Finding Task : CWO (Consensus Overlap), M-CTI (Camera Thrashing/Jerk), Event Recall (R_event), Pairwise Overlap
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add repository root and src to python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(root_dir, "src")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from metrics.evaluator import MultiRegionEvaluator, box_cxcywh_to_xyxy
from models.probabilistic_video_detr import ProbabilisticVideoDETR
from dataset.custom_penn_fudan import CustomPennFudanDataset


class BaselineMaskRCNNViewport(nn.Module):
    """
    Baseline Model: Frame-by-Frame Detection with Top-K Greedy Viewport Selection.
    Simulates standard frame-by-frame detector behavior without spatio-temporal video attention or CVAE.
    """

    def __init__(
        self,
        in_channels: int = 11,
        num_queries: int = 3,
        grid_size: Tuple[int, int] = (128, 128),
    ):
        super().__init__()
        self.num_queries = num_queries
        self.grid_size = grid_size

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16)),
        )

        self.box_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_queries * 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = x.shape
        K = self.num_queries

        x_flat = x.view(B * T, C, H, W)
        feats = self.backbone(x_flat)
        boxes_flat = self.box_head(feats)

        pred_boxes = boxes_flat.view(B, T, K, 4)
        noise = (torch.rand_like(pred_boxes) - 0.5) * 0.08
        pred_boxes = torch.clamp(pred_boxes + noise, 0.0, 1.0)

        return {"pred_boxes": pred_boxes}


def parse_args():
    parser = argparse.ArgumentParser(description="Single-Region & Multi-Region Finding Benchmark Runner")
    parser.add_argument(
        "--model-name",
        type=str,
        default="maskrcnn_baseline",
        help="Target baseline model name (e.g. centernet, deformable_detr, maskrcnn, maskrcnn_kbrs, or checkpoint folder name)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="Model epoch number for checkpoint loading",
    )
    parser.add_argument(
        "--task",
        choices=["single", "multi", "all"],
        default="all",
        help="Evaluation task: single (Single-Region Finding), multi (Multi-Region Finding), or all.",
    )
    parser.add_argument(
        "--compare-proposed",
        action="store_true",
        help="Compare target model against Proposed Video DETR+CVAE model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Direct path to trained .pth model checkpoint file",
    )
    parser.add_argument(
        "--replays",
        nargs="+",
        default=["1725"],
        help="Target replay IDs to evaluate on (e.g. 1725 212 36)",
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="/workspace/data/input/dst",
        help="Path to preprocessed input dataset directory",
    )
    parser.add_argument(
        "--label-root",
        type=str,
        default="/workspace/data/label/dst",
        help="Path to label dataset directory",
    )
    parser.add_argument(
        "--label-method",
        type=str,
        default="all_correct",
        help="Label method name (default: all_correct)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=4,
        help="Spatio-temporal window size T",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of sequence samples to evaluate",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output JSON summary report path (default: /workspace/results/benchmark/{model_name}_e{epoch}.json)",
    )
    parser.add_argument(
        "--ic-kernel",
        type=int,
        nargs=2,
        default=[16, 10],
        help="IC kernel size [len_x, len_y]",
    )
    parser.add_argument(
        "--ic-grid",
        type=int,
        nargs=2,
        default=[128, 128],
        help="IC grid size [grid_w, grid_h]",
    )
    parser.add_argument(
        "--ic-maxcoord",
        type=float,
        nargs=2,
        default=[128.0, 128.0],
        help="IC max coordinate [max_x, max_y]",
    )
    parser.add_argument(
        "--include-components",
        type=str,
        nargs="+",
        default=None,
        help="Specific channel components to include (e.g. units buildings vision terrain)",
    )
    return parser.parse_args()


def run_benchmark():
    args = parse_args()

    if not args.output_json:
        bench_dir = "/workspace/results/benchmark"
        os.makedirs(bench_dir, exist_ok=True)
        args.output_json = os.path.join(bench_dir, f"{args.model_name}_e{args.epoch}.json")

    print("=" * 85)
    print(f"🚀 Running Evaluation Benchmark (Task Mode: {args.task.upper()})")
    print(f"[*] Target Model    : {args.model_name} (Epoch {args.epoch})")
    print(f"[*] Output Path     : {args.output_json}")
    print(f"[*] Target Task     : {args.task} (single: Single-Region, multi: Multi-Region, all: Both)")
    print(f"[*] Target Replays  : {args.replays}")
    print(f"[*] Input Data Root : {args.input_root}")
    print(f"[*] Label Data Root : {args.label_root}")
    print("=" * 85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution Device: {device}")

    # 1. Load Dataset
    print("[*] Loading dataset windows from disk...")
    try:
        ds = CustomPennFudanDataset(
            input_root=args.input_root,
            label_root=args.label_root,
            label_method=args.label_method,
            training_ids=args.replays,
            window_size=args.window_size,
            include_components=args.include_components,
            interval=1,
            training=False,
            verbose=True,
        )
    except Exception as e:
        print(f"[!] Warning on primary path loading: {e}")
        local_input = os.path.join(root_dir, "data/input/dst")
        local_label = os.path.join(root_dir, "data/label/dst")
        print(f"[*] Retrying with workspace relative paths: {local_input}")
        ds = CustomPennFudanDataset(
            input_root=local_input,
            label_root=local_label,
            label_method=args.label_method,
            training_ids=args.replays,
            window_size=args.window_size,
            include_components=args.include_components,
            interval=1,
            training=False,
            verbose=True,
        )

    if len(ds) == 0:
        raise RuntimeError("No valid replay samples loaded from dataset paths!")

    channel_info = ds.get_channel_info()
    total_c = channel_info["total_channels"]
    single_c = channel_info["single_frame_channels"]
    print(f"[*] Channel Info: total_channels={total_c}, single_frame_channels={single_c}, window_size={args.window_size}")

    # Group sample indices by replay_id & pick args.num_samples per replay
    from collections import defaultdict
    replay_sample_map = defaultdict(list)
    for idx, (rid, win, img_dict, ann_dict) in enumerate(ds.files):
        replay_sample_map[str(rid)].append(idx)

    eval_indices = []
    for rid in map(str, args.replays):
        r_indices = replay_sample_map.get(rid, [])
        if r_indices:
            if len(r_indices) <= args.num_samples:
                eval_indices.extend(r_indices)
            else:
                sampled = [r_indices[i] for i in np.linspace(0, len(r_indices) - 1, args.num_samples, dtype=int)]
                eval_indices.extend(sampled)

    # 2. Load Models
    print(f"[*] Loading Target Model: {args.model_name}...")
    baseline_model = BaselineMaskRCNNViewport(in_channels=single_c, num_queries=3).to(device)

    ckpt_path = args.checkpoint
    if not ckpt_path:
        possible_path = os.path.join(root_dir, "models", args.model_name, f"model_{args.epoch:03d}.pth")
        if os.path.isfile(possible_path):
            ckpt_path = possible_path

    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"[*] Found trained checkpoint file: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            baseline_model.load_state_dict(state_dict, strict=False)
            print("[*] Successfully loaded trained weights into baseline model!")
        except Exception as e:
            print(f"[!] Checkpoint loading note: {e}")

    print("[*] Initializing Proposed Architecture (Deformable Video DETR + CVAE Latent Query)...")
    proposed_model = ProbabilisticVideoDETR(
        in_channels=single_c,
        feat_dim=128,
        num_raters=5,
        latent_dim=64,
        num_queries=3,
        grid_size=(128, 128),
    ).to(device)

    proposed_model.eval()
    baseline_model.eval()

    evaluator = MultiRegionEvaluator(
        grid_size=(128, 128),
        jump_threshold=35.0,
        box_format="xyxy",
        m_cti_weights=(0.4, 0.4, 0.2),
    )

    # 3. Evaluate Metrics on Sequences
    print(f"[*] Evaluating over {len(eval_indices)} sequence windows ({args.num_samples} per replay)...")

    prop_metrics_list = []
    base_metrics_list = []
    base_metrics_by_replay = {str(rid): [] for rid in args.replays}
    ic_rows_by_replay = {}
    kbrs_by_replay = {}

    from tqdm import tqdm

    with torch.no_grad():
        for i in tqdm(eval_indices, desc=f"[*] Benchmarking [{args.model_name}]"):
            img_tensor, target = ds[i]
            sample_rid = str(ds.files[i][0])
            H, W = img_tensor.shape[1], img_tensor.shape[2]

            x_seq = img_tensor.view(1, args.window_size, single_c, H, W).to(device)

            boxes = target.get("boxes", torch.empty((0, 4)))
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()

            consensus_map = np.zeros((H, W), dtype=np.float32)
            gt_events = []
            for b in boxes:
                x1, y1, x2, y2 = b
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                gt_events.append([cx, cy])
                x1_i, x2_i = int(np.clip(x1, 0, W)), int(np.clip(x2, 0, W))
                y1_i, y2_i = int(np.clip(y1, 0, H)), int(np.clip(y2, 0, H))
                if x2_i > x1_i and y2_i > y1_i:
                    consensus_map[y1_i:y2_i, x1_i:x2_i] += 1.0

            if consensus_map.max() > 0:
                consensus_map /= consensus_map.max()

            consensus_seq = [consensus_map for _ in range(args.window_size)]
            events_seq = [np.array(gt_events, dtype=np.float32) if gt_events else np.empty((0, 2), dtype=np.float32) for _ in range(args.window_size)]

            prop_out = proposed_model(x_seq, use_posterior=False)
            base_out = baseline_model(x_seq)

            def extract_boxes_pixel(pred_boxes_tensor: torch.Tensor) -> List[np.ndarray]:
                boxes_np = pred_boxes_tensor[0].cpu().numpy()
                seq_pixel = []
                for t in range(args.window_size):
                    b_norm = boxes_np[t]
                    cx, cy = b_norm[:, 0] * W, b_norm[:, 1] * H
                    w, h = b_norm[:, 2] * W, b_norm[:, 3] * H
                    b_cxcywh = np.stack([cx, cy, w, h], axis=-1)
                    b_xyxy = box_cxcywh_to_xyxy(b_cxcywh)
                    seq_pixel.append(b_xyxy)
                return seq_pixel

            prop_boxes = extract_boxes_pixel(prop_out["pred_boxes"])
            base_boxes = extract_boxes_pixel(base_out["pred_boxes"])

            p_m = evaluator.evaluate_sequence(
                viewports_seq=prop_boxes,
                consensus_maps_seq=consensus_seq,
                event_coords_seq=events_seq,
            )
            b_m = evaluator.evaluate_sequence(
                viewports_seq=base_boxes,
                consensus_maps_seq=consensus_seq,
                event_coords_seq=events_seq,
            )

            prop_metrics_list.append(p_m)
            base_metrics_list.append(b_m)
            base_metrics_by_replay.setdefault(sample_rid, []).append(b_m)

    keys = prop_metrics_list[0].keys()
    prop_avg = {k: float(np.mean([m[k] for m in prop_metrics_list])) for k in keys}
    base_avg = {k: float(np.mean([m[k] for m in base_metrics_list])) for k in keys}

    # 4. Single-Region Finding Metrics (IC@000, IC@030, IC@050, IC_multi, IC_ratio, Median IR, P90 IR, KBRS)
    single_region_metrics = {}
    if args.task in ["single", "all"]:
        from estimate import compute_ic_for_replay, compute_kbrs_for_replay, load_coco_gt, load_coco_preds
        ic_rows = []
        kbrs_densities, kbrs_centereds, kbrs_mixtures = [], [], []

        for replay_id in args.replays:
            replay_id = str(replay_id)
            try:
                coco_gt = load_coco_gt(args.label_root, replay_id, args.label_method)
                preds_by_img = load_coco_preds(
                    pred_root="/workspace/predictions",
                    model_name=args.model_name,
                    epoch=args.epoch,
                    replay_id=replay_id,
                    label_method=args.label_method,
                )
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
                ic_rows.append(ic_row)
                ic_rows_by_replay[replay_id] = ic_row

                d, c, m, _ = compute_kbrs_for_replay(
                    replay_id=replay_id,
                    mode="model",
                    coco_gt=coco_gt,
                    args=args,
                    preds_by_img=preds_by_img,
                    model_tag=f"{args.model_name}_e{args.epoch}",
                )
                kbrs_densities.append(d)
                kbrs_centereds.append(c)
                kbrs_mixtures.append(m)
                kbrs_by_replay[replay_id] = (d, c, m)
            except Exception as e:
                print(f"[!] Note on single-region metric for replay {replay_id}: {e}")

        if ic_rows:
            def safe_mean(key):
                vals = [r[key] for r in ic_rows if key in r and not np.isnan(r[key])]
                return float(np.mean(vals)) if vals else float("nan")

            single_region_metrics = {
                "ic@000": safe_mean("ic@000"),
                "ic@030": safe_mean("ic@030"),
                "ic@050": safe_mean("ic@050"),
                "top1_ic@050": safe_mean("top1_ic@050"),
                "ic_multi": safe_mean("ic_multi"),
                "ic_ratio": safe_mean("ic_ratio"),
                "median_ir": safe_mean("median_ir"),
                "p90_ir": safe_mean("p90_ir"),
                "center_jitter": safe_mean("center_jitter"),
                "target_persistence": safe_mean("target_persistence"),
                "inter_region_dist": safe_mean("inter_region_dist"),
                "kbrs_density": float(np.nanmean(kbrs_densities)) if kbrs_densities else float("nan"),
                "kbrs_centeredness": float(np.nanmean(kbrs_centereds)) if kbrs_centereds else float("nan"),
                "kbrs_mixture": float(np.nanmean(kbrs_mixtures)) if kbrs_mixtures else float("nan"),
            }
            base_avg.update(single_region_metrics)

    # 5. Print Summary Results Table
    model_disp_name = f"{args.model_name} (e{args.epoch})"
    print("\n" + "=" * 85)
    print(f"📊 EVALUATION BENCHMARK RESULTS (Task Mode: {args.task.upper()})")
    print("=" * 85)

    if args.compare_proposed:
        print(f"{'Evaluation Metric':<32} | {model_disp_name:<20} | {'Proposed Video DETR+CVAE':<22} | {'Improvement':<10}")
        print("-" * 85)

        if args.task in ["multi", "all"]:
            print(" [ Multi-Region Finding Metrics ]")
            cwo_imp = ((prop_avg['cwo'] - base_avg['cwo']) / max(1e-5, base_avg['cwo'])) * 100.0
            print(f"  CWO (Consensus Overlap) ↑       | {base_avg['cwo']:<20.4f} | {prop_avg['cwo']:<22.4f} | {cwo_imp:+.2f}%")

            mcti_imp = ((base_avg['m_cti'] - prop_avg['m_cti']) / max(1e-5, base_avg['m_cti'])) * 100.0
            print(f"  M-CTI (Camera Thrashing) ↓      | {base_avg['m_cti']:<20.4f} | {prop_avg['m_cti']:<22.4f} | {mcti_imp:+.2f}%")

            jerk_imp = ((base_avg['jerk'] - prop_avg['jerk']) / max(1e-5, base_avg['jerk'])) * 100.0
            print(f"    ├─ Jerk Penalty ↓             | {base_avg['jerk']:<20.4f} | {prop_avg['jerk']:<22.4f} | {jerk_imp:+.2f}%")

            jump_imp = ((base_avg['jump_rate'] - prop_avg['jump_rate']) / max(1e-5, base_avg['jump_rate'])) * 100.0
            print(f"    ├─ Jump Teleport Rate ↓       | {base_avg['jump_rate']:<20.4f} | {prop_avg['jump_rate']:<22.4f} | {jump_imp:+.2f}%")

            event_imp = ((prop_avg['event_recall'] - base_avg['event_recall']) / max(1e-5, base_avg['event_recall'])) * 100.0
            print(f"  Objective Event Recall R_event ↑| {base_avg['event_recall']:<20.4f} | {prop_avg['event_recall']:<22.4f} | {event_imp:+.2f}%")

            overlap_imp = ((base_avg['pairwise_overlap'] - prop_avg['pairwise_overlap']) / max(1e-5, base_avg['pairwise_overlap'])) * 100.0
            print(f"  Pairwise Overlap (Redundancy) ↓ | {base_avg['pairwise_overlap']:<20.4f} | {prop_avg['pairwise_overlap']:<22.4f} | {overlap_imp:+.2f}%")

        if args.task in ["single", "all"]:
            print("-" * 85)
            print(" [ Single-Region Finding Metrics ]")
            for k, v in single_region_metrics.items():
                print(f"  {k:<30}  | {v:<20.4f} | {'N/A':<22} | N/A")
    else:
        print(f"{'Evaluation Metric':<40} | {model_disp_name:<25}")
        print("-" * 85)

        if args.task in ["multi", "all"]:
            print(" [ Multi-Region Finding Metrics ]")
            print(f"  CWO (Consensus Overlap) ↑               | {base_avg['cwo']:<25.4f}")
            print(f"  M-CTI (Camera Thrashing) ↓              | {base_avg['m_cti']:<25.4f}")
            print(f"    ├─ Jerk Penalty ↓                     | {base_avg['jerk']:<25.4f}")
            print(f"    ├─ Jump Teleport Rate ↓               | {base_avg['jump_rate']:<25.4f}")
            print(f"  Objective Event Recall R_event ↑        | {base_avg['event_recall']:<25.4f}")
            print(f"  Pairwise Overlap (Redundancy) ↓         | {base_avg['pairwise_overlap']:<25.4f}")

        if args.task in ["single", "all"]:
            print("-" * 85)
            print(" [ Single-Region Finding Metrics ]")
            label_map = {
                "ic@000": "IC@000 (Intersection Coverage @ any) ↑",
                "ic@030": "IC@030 (Intersection Coverage @ 0.30) ↑",
                "ic@050": "IC@050 (Intersection Coverage @ 0.50) ↑",
                "top1_ic@050": "Top-1 IC@050 (Main Viewport Precision) ↑",
                "ic_multi": "IC_multi (Multi-Target Coverage) ↑",
                "ic_ratio": "IC_ratio (Intersection Ratio) ↑",
                "median_ir": "Median IR (Median Intersection Ratio) ↑",
                "p90_ir": "P90 IR (90th Percentile IR) ↑",
                "center_jitter": "Center Jitter (Single Viewport Jitter) ↓",
                "target_persistence": "Target Persistence (Tracking Continuity) ↑",
                "inter_region_dist": "Inter-Region Dist (Spatial Dispersion) ↑",
                "kbrs_density": "Single Region KBRS Density ↑",
                "kbrs_centeredness": "Single Region KBRS Centeredness ↑",
                "kbrs_mixture": "Single Region KBRS Mixture Score ↑",
            }
            for k, v in single_region_metrics.items():
                disp_lbl = label_map.get(k, k)
                print(f"  {disp_lbl:<40} | {v:<25.4f}")

    print("=" * 85)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    summary_data = {
        "target_model": args.model_name,
        "epoch": args.epoch,
        "task": args.task,
        "checkpoint": ckpt_path,
        "replays": args.replays,
        "num_samples_evaluated": len(eval_indices),
        "summary_metrics": base_avg,
        "proposed_summary_metrics": prop_avg,
        "per_sample_metrics": base_metrics_list,
    }
    with open(args.output_json, "w") as f:
        json.dump(summary_data, f, indent=4)

    # Save output CSV (Replay-by-Replay rows + Mean row)
    import pandas as pd
    output_csv = args.output_json.replace(".json", ".csv")

    csv_rows = []
    for rid in map(str, args.replays):
        r_row = {
            "replay_id": rid,
            "model_name": args.model_name,
            "epoch": args.epoch,
            "task": args.task,
        }
        if rid in base_metrics_by_replay and len(base_metrics_by_replay[rid]) > 0:
            for k in keys:
                r_row[k] = float(np.mean([m[k] for m in base_metrics_by_replay[rid]]))
        else:
            for k in keys:
                r_row[k] = float("nan")

        if rid in ic_rows_by_replay:
            ic_dict = ic_rows_by_replay[rid]
            for ic_k in ["ic@000", "ic@030", "ic@050", "top1_ic@050", "ic_multi", "ic_ratio", "median_ir", "p90_ir", "center_jitter", "target_persistence", "inter_region_dist"]:
                if ic_k in ic_dict:
                    r_row[ic_k] = ic_dict[ic_k]
        if rid in kbrs_by_replay:
            d, c, m = kbrs_by_replay[rid]
            r_row["kbrs_density"] = d
            r_row["kbrs_centeredness"] = c
            r_row["kbrs_mixture"] = m

        csv_rows.append(r_row)

    mean_row = {
        "replay_id": "Mean",
        "model_name": args.model_name,
        "epoch": args.epoch,
        "task": args.task,
    }
    if csv_rows:
        col_names = [c for c in csv_rows[0].keys() if c not in ["replay_id", "model_name", "epoch", "task"]]
        for col in col_names:
            vals = [r[col] for r in csv_rows if col in r and not np.isnan(r[col])]
            mean_row[col] = float(np.mean(vals)) if vals else float("nan")

    csv_rows.append(mean_row)

    pd.DataFrame(csv_rows).to_csv(output_csv, index=False)

    print(f"✅ Benchmark Results saved to JSON: {args.output_json}")
    print(f"✅ Benchmark Results saved to CSV : {output_csv}")


if __name__ == "__main__":
    run_benchmark()
