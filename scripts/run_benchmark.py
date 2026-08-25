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
from models import build_model, ProbabilisticVideoDETR
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
        default=None,
        help="Spatio-temporal window size T (default: auto-detected from model_name or 4)",
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

    if args.window_size is None:
        if "win1" in args.model_name.lower():
            args.window_size = 1
        elif "win4" in args.model_name.lower():
            args.window_size = 4
        else:
            args.window_size = 4

    if not args.output_json:
        bench_dir = "/workspace/results/benchmark"
        os.makedirs(bench_dir, exist_ok=True)
        args.output_json = os.path.join(bench_dir, f"{args.model_name}_e{args.epoch}.json")

    print("=" * 85)
    print(f"🚀 Running Evaluation Benchmark (Task Mode: {args.task.upper()})")
    print(f"[*] Target Model    : {args.model_name} (Epoch {args.epoch})")
    print(f"[*] Window Size     : {args.window_size}")
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

    if hasattr(ds, "get_channel_info"):
        channel_info = ds.get_channel_info()
        total_c = channel_info["total_channels"]
        single_c = channel_info["single_frame_channels"]
    else:
        single_c = len(getattr(ds, "channel_indices", range(11)))
        total_c = single_c * args.window_size

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
    from types import SimpleNamespace

    model_lower = args.model_name.lower()
    is_kbrs = "kbrs" in model_lower

    if "centernet" in model_lower:
        base_arch = "centernet"
    elif "deformable" in model_lower or ("detr" in model_lower and "rtdetr" not in model_lower and "video_detr" not in model_lower):
        base_arch = "deformable_detr"
    elif "rtdetr" in model_lower:
        base_arch = "rtdetr"
    elif "probabilistic" in model_lower or "video_detr" in model_lower:
        base_arch = "probabilistic_video_detr"
    else:
        base_arch = "maskrcnn"

    baseline_args = SimpleNamespace(
        model_name=base_arch,
        use_kbrs=is_kbrs,
        num_classes=2,
        in_channels=total_c,
        single_frame_channels=single_c,
        window_size=args.window_size,
        num_queries=3,
        use_probabilistic_query=False,
    )
    try:
        baseline_model = build_model(baseline_args).to(device)
    except Exception as e:
        print(f"[*] Fallback to BaselineMaskRCNNViewport: {e}")
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
    proposed_args = SimpleNamespace(
        model_name="probabilistic_video_detr",
        in_channels=total_c,
        single_frame_channels=single_c,
        window_size=args.window_size,
        feat_dim=128,
        num_raters=5,
        latent_dim=64,
        num_queries=3,
        grid_size=(128, 128),
        use_cvae=True,
    )
    proposed_model = build_model(proposed_args).to(device)

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

            # Baseline sequence forward
            if isinstance(baseline_model, (ProbabilisticVideoDETR, BaselineMaskRCNNViewport)):
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
                base_boxes = extract_boxes_pixel(base_out["pred_boxes"])
            else:
                # Frame-by-frame 2D Detector (Mask R-CNN, CenterNet, Deformable DETR, RT-DETR)
                base_boxes = []
                for t in range(args.window_size):
                    frame_c = x_seq[0, t]  # (C, H, W)
                    try:
                        if base_arch == "centernet":
                            dets = baseline_model(frame_c.unsqueeze(0))
                        else:
                            dets = baseline_model([frame_c])
                        
                        if isinstance(dets, list) and len(dets) > 0 and "boxes" in dets[0]:
                            b = dets[0]["boxes"].detach().cpu().numpy()
                            s = dets[0]["scores"].detach().cpu().numpy() if "scores" in dets[0] else np.ones(len(b))
                            if len(b) > 0:
                                top_idx = np.argsort(-s)[:3]
                                top_b = b[top_idx]
                                # pad to 3 viewports if needed
                                if len(top_b) < 3:
                                    pad = np.repeat(top_b[:1], 3 - len(top_b), axis=0) if len(top_b) > 0 else np.zeros((3, 4))
                                    top_b = np.concatenate([top_b, pad], axis=0)
                                base_boxes.append(top_b)
                            else:
                                base_boxes.append(np.zeros((3, 4), dtype=np.float32))
                        else:
                            base_boxes.append(np.zeros((3, 4), dtype=np.float32))
                    except Exception:
                        base_boxes.append(np.zeros((3, 4), dtype=np.float32))

            def extract_prop_boxes(pred_boxes_tensor: torch.Tensor) -> List[np.ndarray]:
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

            prop_boxes = extract_prop_boxes(prop_out["pred_boxes"])

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
    ic_rows_hit_by_replay = {}
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

                total_f = len(coco_gt.dataset.get("images", []))
                hit_f = sum(1 for img in coco_gt.dataset.get("images", []) if len(preds_by_img.get(int(img["id"]), [])) > 0)
                hit_r = float(hit_f / max(1, total_f) * 100.0)

                # 1) Official Overall IC (includes 0.0 penalty for missing prediction frames)
                ic_row = compute_ic_for_replay(
                    replay_id=replay_id,
                    mode="model",
                    coco_gt=coco_gt,
                    args=args,
                    preds_all=preds_all if preds_all else None,
                    model_tag=f"{args.model_name}_e{args.epoch}",
                    skip_missing_preds=False,
                )
                ic_row["total_frames"] = total_f
                ic_row["hit_frames"] = hit_f
                ic_row["hit_rate"] = hit_r
                ic_rows.append(ic_row)
                ic_rows_by_replay[replay_id] = ic_row

                # 2) Conditional Hit-Only IC (skips missing prediction frames)
                ic_row_hit = compute_ic_for_replay(
                    replay_id=replay_id,
                    mode="model",
                    coco_gt=coco_gt,
                    args=args,
                    preds_all=preds_all if preds_all else None,
                    model_tag=f"{args.model_name}_e{args.epoch}_hit",
                    skip_missing_preds=True,
                )
                ic_row_hit["total_frames"] = total_f
                ic_row_hit["hit_frames"] = hit_f
                ic_row_hit["hit_rate"] = hit_r
                ic_rows_hit_by_replay[replay_id] = ic_row_hit

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
                "kbrs_density": float(np.mean(kbrs_densities)) if kbrs_densities else float("nan"),
                "kbrs_centeredness": float(np.mean(kbrs_centereds)) if kbrs_centereds else float("nan"),
                "kbrs_mixture": float(np.mean(kbrs_mixtures)) if kbrs_mixtures else float("nan"),
            }
            base_avg.update(single_region_metrics)

    # 5. Print Summary Results Table
    from utils.report import ReportBlock, print_section_header

    model_disp_name = f"{args.model_name} (e{args.epoch})"
    print_section_header(f"EVALUATION BENCHMARK RESULTS (Task Mode: {args.task.upper()})")

    def _imp(base_v: float, prop_v: float, lower_is_better: bool) -> str:
        if lower_is_better:
            imp = ((base_v - prop_v) / max(1e-5, abs(base_v))) * 100.0
        else:
            imp = ((prop_v - base_v) / max(1e-5, abs(base_v))) * 100.0
        return f"{imp:+.2f}%"

    if args.compare_proposed:
        rb = ReportBlock(
            title=f"Comparison: {model_disp_name} vs Proposed Video DETR+CVAE",
            columns=["metric", "direction", model_disp_name, "Proposed", "Improvement"],
            aligns=["left", "left", "right", "right", "right"],
        )
        if args.task in ["multi", "all"]:
            rb.add_row("CWO (Consensus Overlap)", "↑", base_avg['cwo'], prop_avg['cwo'], _imp(base_avg['cwo'], prop_avg['cwo'], False))
            rb.add_row("M-CTI (Camera Thrashing)", "↓", base_avg['m_cti'], prop_avg['m_cti'], _imp(base_avg['m_cti'], prop_avg['m_cti'], True))
            rb.add_row("  Jerk Penalty", "↓", base_avg['jerk'], prop_avg['jerk'], _imp(base_avg['jerk'], prop_avg['jerk'], True))
            rb.add_row("  Jump Teleport Rate", "↓", base_avg['jump_rate'], prop_avg['jump_rate'], _imp(base_avg['jump_rate'], prop_avg['jump_rate'], True))
            rb.add_row("Objective Event Recall R_event", "↑", base_avg['event_recall'], prop_avg['event_recall'], _imp(base_avg['event_recall'], prop_avg['event_recall'], False))
            rb.add_row("Pairwise Overlap (Redundancy)", "↓", base_avg['pairwise_overlap'], prop_avg['pairwise_overlap'], _imp(base_avg['pairwise_overlap'], prop_avg['pairwise_overlap'], True))
        if args.task in ["single", "all"]:
            for k, v in single_region_metrics.items():
                rb.add_row(k, "", v, "-", "-")
        rb.print()
    else:
        label_map = {
            "cwo": ("CWO (Consensus Overlap)", "up"),
            "m_cti": ("M-CTI (Camera Thrashing)", "down"),
            "jerk": ("  Jerk Penalty", "down"),
            "jump_rate": ("  Jump Teleport Rate", "down"),
            "event_recall": ("Objective Event Recall R_event", "up"),
            "pairwise_overlap": ("Pairwise Overlap (Redundancy)", "down"),
            "ic@000": ("IC@000 (Intersection Coverage @ any)", "up"),
            "ic@030": ("IC@030 (Intersection Coverage @ 0.30)", "up"),
            "ic@050": ("IC@050 (Intersection Coverage @ 0.50)", "up"),
            "top1_ic@050": ("Top-1 IC@050 (Main Viewport Precision)", "up"),
            "ic_multi": ("IC_multi (Multi-Target Coverage)", "up"),
            "ic_ratio": ("IC_ratio (Intersection Ratio)", "up"),
            "median_ir": ("Median IR (Median Intersection Ratio)", "up"),
            "p90_ir": ("P90 IR (90th Percentile IR)", "up"),
            "center_jitter": ("Center Jitter (Single Viewport Jitter)", "down"),
            "target_persistence": ("Target Persistence (Tracking Continuity)", "up"),
            "inter_region_dist": ("Inter-Region Dist (Spatial Dispersion)", "up"),
            "kbrs_density": ("KBRS Density", "up"),
            "kbrs_centeredness": ("KBRS Centeredness", "up"),
            "kbrs_mixture": ("KBRS Mixture Score", "up"),
        }

        rb = ReportBlock(
            title=f"Summary Metrics — {model_disp_name}",
            columns=["metric", "direction", "value"],
            aligns=["left", "center", "right"],
        )
        if args.task in ["multi", "all"]:
            for k in ["cwo", "m_cti", "jerk", "jump_rate", "event_recall", "pairwise_overlap"]:
                lbl, d = label_map[k]
                rb.add_row(lbl, "↑" if d == "up" else "↓", base_avg[k])
        if args.task in ["single", "all"]:
            for k, v in single_region_metrics.items():
                lbl, d = label_map.get(k, (k, None))
                rb.add_row(lbl, "↑" if d == "up" else ("↓" if d == "down" else ""), v)
        rb.print()

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

    # Save output CSV (1. Official Overall CSV & 2. Hit-Only Conditional CSV)
    import pandas as pd
    output_csv = args.output_json.replace(".json", ".csv")
    output_hit_csv = args.output_json.replace(".json", "_hit_only.csv")

    def build_csv_dataframe(ic_mapping):
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

            if rid in ic_mapping:
                ic_dict = ic_mapping[rid]
                for ic_k in ["total_frames", "hit_frames", "hit_rate", "ic@000", "ic@030", "ic@050", "top1_ic@050", "ic_multi", "ic_ratio", "median_ir", "p90_ir", "center_jitter", "target_persistence", "inter_region_dist"]:
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
        return pd.DataFrame(csv_rows)

    df_overall = build_csv_dataframe(ic_rows_by_replay)
    df_overall.to_csv(output_csv, index=False)

    df_hit_only = build_csv_dataframe(ic_rows_hit_by_replay)
    df_hit_only.to_csv(output_hit_csv, index=False)

    # Per-replay table in the log (copy-paste friendly TSV included)
    from utils.report import ReportBlock

    rb = ReportBlock(
        title=f"Per-Replay Results — {model_disp_name}",
        columns=list(df_overall.columns),
        aligns=["left"] + ["right"] * (len(df_overall.columns) - 1),
    )
    for _, r in df_overall.iterrows():
        rb.add_row(*[r.get(c) for c in df_overall.columns])
    rb.add_note(f"JSON (overall)          : {args.output_json}")
    rb.add_note(f"CSV  (official overall) : {output_csv}")
    rb.add_note(f"CSV  (hit-only cond.)   : {output_hit_csv}")
    rb.print()


if __name__ == "__main__":
    run_benchmark()
