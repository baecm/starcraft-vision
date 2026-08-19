#!/usr/bin/env python3
"""
Real Dataset Disk I/O Benchmark Script:
Loads real preprocessed StarCraft replay state tensors (from data/input/dst)
and real COCO annotations/pickle labels (from data/label/dst) using CustomPennFudanDataset / load_data,
runs inference on Proposed ProbabilisticVideoDETR vs Baseline Detector,
and computes 3D Evaluation Suite metrics (CWO, M-CTI, Event Recall, Pairwise Overlap) on real dataset I/O.
"""

from __future__ import annotations

import os
import sys
import json
import time
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

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
from scripts.run_benchmark import BaselineMaskRCNNViewport


def run_real_evaluation(
    input_root: str = "/workspace/data/input/dst",
    label_root: str = "/workspace/data/label/dst",
    replays: List[str] = None,
    label_method: str = "all_correct",
    window_size: int = 4,
    batch_size: int = 2,
    output_json: str = "/workspace/results/3d_real_evaluation_results.json",
):
    print("=" * 85)
    print("🚀 Running Real Dataset Disk I/O 3D Evaluation Suite Benchmark")
    print("=" * 85)

    if replays is None:
        replays = ["1725"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution Device: {device}")
    print(f"[*] Input Data Root : {input_root}")
    print(f"[*] Label Data Root : {label_root}")
    print(f"[*] Target Replays  : {replays}")

    # 1. Instantiate CustomPennFudanDataset with Real Disk I/O
    print("[*] Loading real dataset windows from disk...")
    try:
        ds = CustomPennFudanDataset(
            input_root=input_root,
            label_root=label_root,
            label_method=label_method,
            training_ids=replays,
            window_size=window_size,
            interval=1,
            indices=list(range(20)),
            training=False,
            verbose=True,
        )
        print(f"[*] Real Dataset Windows Loaded: {len(ds)} samples.")
    except Exception as e:
        print(f"[!] Exception during dataset initialization: {e}")
        # Fallback to local workspace relative paths if needed
        local_input = os.path.join(root_dir, "data/input/dst")
        local_label = os.path.join(root_dir, "data/label/dst")
        print(f"[*] Retrying with local paths: input={local_input}, label={local_label}")
        ds = CustomPennFudanDataset(
            input_root=local_input,
            label_root=local_label,
            label_method=label_method,
            training_ids=replays,
            window_size=window_size,
            interval=1,
            training=False,
            verbose=True,
        )
        print(f"[*] Real Dataset Windows Loaded: {len(ds)} samples.")

    if len(ds) == 0:
        raise RuntimeError("No valid replay samples loaded from dataset paths!")

    # Inspect input channel dimensions
    sample_img, sample_target = ds[0]
    in_channels = sample_img.shape[0]  # total channels (C_single * window_size)
    single_c = max(1, in_channels // window_size)
    print(f"[*] Channel Info: total_channels={in_channels}, single_frame_channels={single_c}, window_size={window_size}")

    # 2. Build Models
    grid_size = (128, 128)
    num_queries = 3  # 1 Main + 2 PIP Viewports

    print("[*] Initializing Models for Real I/O Evaluation...")
    proposed_model = ProbabilisticVideoDETR(
        in_channels=single_c,
        feat_dim=128,
        num_raters=5,
        latent_dim=64,
        num_queries=num_queries,
        grid_size=grid_size,
    ).to(device)

    baseline_model = BaselineMaskRCNNViewport(
        in_channels=single_c,
        num_queries=num_queries,
        grid_size=grid_size,
    ).to(device)

    proposed_model.eval()
    baseline_model.eval()

    evaluator = MultiRegionEvaluator(
        grid_size=grid_size,
        jump_threshold=35.0,
        box_format="xyxy",
        m_cti_weights=(0.4, 0.4, 0.2),
    )

    # 3. Process Real Sequences through DataLoader
    num_eval_samples = min(5, len(ds))
    print(f"[*] Evaluating on {num_eval_samples} real disk sequence windows...")

    prop_batch_metrics = []
    base_batch_metrics = []

    with torch.no_grad():
        for i in range(num_eval_samples):
            img_tensor, target = ds[i]
            # img_tensor shape: (C_total, H, W) where C_total = window_size * C_single
            H, W = img_tensor.shape[1], img_tensor.shape[2]

            # Reshape to (B=1, T=window_size, C=single_c, H, W)
            x_seq = img_tensor.view(1, window_size, single_c, H, W).to(device)

            # Build consensus map and GT event points from real target annotations
            boxes = target.get("boxes", torch.empty((0, 4)))
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()

            # Construct consensus density map from GT boxes
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

            consensus_seq = [consensus_map for _ in range(window_size)]
            events_seq = [np.array(gt_events, dtype=np.float32) if gt_events else np.empty((0, 2), dtype=np.float32) for _ in range(window_size)]

            # Forward passes
            prop_out = proposed_model(x_seq, use_posterior=False)
            base_out = baseline_model(x_seq)

            # Convert predictions
            def extract_boxes_pixel(pred_boxes_tensor: torch.Tensor) -> List[np.ndarray]:
                boxes_np = pred_boxes_tensor[0].cpu().numpy()  # (T, K, 4)
                seq_pixel = []
                for t in range(window_size):
                    b_norm = boxes_np[t]
                    cx, cy = b_norm[:, 0] * W, b_norm[:, 1] * H
                    w, h = b_norm[:, 2] * W, b_norm[:, 3] * H
                    b_cxcywh = np.stack([cx, cy, w, h], axis=-1)
                    b_xyxy = box_cxcywh_to_xyxy(b_cxcywh)
                    seq_pixel.append(b_xyxy)
                return seq_pixel

            prop_boxes_seq = extract_boxes_pixel(prop_out["pred_boxes"])
            base_boxes_seq = extract_boxes_pixel(base_out["pred_boxes"])

            # 3D Evaluation Suite per real sequence
            p_metrics = evaluator.evaluate_sequence(
                viewports_seq=prop_boxes_seq,
                consensus_maps_seq=consensus_seq,
                event_coords_seq=events_seq,
            )
            b_metrics = evaluator.evaluate_sequence(
                viewports_seq=base_boxes_seq,
                consensus_maps_seq=consensus_seq,
                event_coords_seq=events_seq,
            )

            prop_batch_metrics.append(p_metrics)
            base_batch_metrics.append(b_metrics)

    # Average metrics over real sequences
    keys = prop_batch_metrics[0].keys()
    prop_avg = {k: float(np.mean([m[k] for m in prop_batch_metrics])) for k in keys}
    base_avg = {k: float(np.mean([m[k] for m in base_batch_metrics])) for k in keys}

    # 4. Display & Save Real Data Benchmark Table
    print("\n" + "=" * 85)
    print("📊 REAL DATASET DISK I/O 3D EVALUATION BENCHMARK RESULTS")
    print("=" * 85)
    print(f"{'3D Evaluation Suite Metric':<32} | {'Baseline Detector':<20} | {'Proposed Video DETR+CVAE':<22} | {'Improvement':<10}")
    print("-" * 85)

    cwo_imp = ((prop_avg['cwo'] - base_avg['cwo']) / max(1e-5, base_avg['cwo'])) * 100.0
    print(f"{'CWO (Consensus Overlap) ↑':<32} | {base_avg['cwo']:<20.4f} | {prop_avg['cwo']:<22.4f} | {cwo_imp:+.2f}%")

    mcti_imp = ((base_avg['m_cti'] - prop_avg['m_cti']) / max(1e-5, base_avg['m_cti'])) * 100.0
    print(f"{'M-CTI (Camera Thrashing) ↓':<32} | {base_avg['m_cti']:<20.4f} | {prop_avg['m_cti']:<22.4f} | {mcti_imp:+.2f}%")

    jerk_imp = ((base_avg['jerk'] - prop_avg['jerk']) / max(1e-5, base_avg['jerk'])) * 100.0
    print(f"{'  ├─ Jerk Penalty ↓':<32} | {base_avg['jerk']:<20.4f} | {prop_avg['jerk']:<22.4f} | {jerk_imp:+.2f}%")

    jump_imp = ((base_avg['jump_rate'] - prop_avg['jump_rate']) / max(1e-5, base_avg['jump_rate'])) * 100.0
    print(f"{'  ├─ Jump Teleport Rate ↓':<32} | {base_avg['jump_rate']:<20.4f} | {prop_avg['jump_rate']:<22.4f} | {jump_imp:+.2f}%")

    event_imp = ((prop_avg['event_recall'] - base_avg['event_recall']) / max(1e-5, base_avg['event_recall'])) * 100.0
    print(f"{'Objective Event Recall R_event ↑':<32} | {base_avg['event_recall']:<20.4f} | {prop_avg['event_recall']:<22.4f} | {event_imp:+.2f}%")

    overlap_imp = ((base_avg['pairwise_overlap'] - prop_avg['pairwise_overlap']) / max(1e-5, base_avg['pairwise_overlap'])) * 100.0
    print(f"{'Pairwise Overlap (Redundancy) ↓':<32} | {base_avg['pairwise_overlap']:<20.4f} | {prop_avg['pairwise_overlap']:<22.4f} | {overlap_imp:+.2f}%")

    print("=" * 85)

    # Save output JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    summary_data = {
        "replays": replays,
        "num_samples_evaluated": num_eval_samples,
        "proposed_metrics": prop_avg,
        "baseline_metrics": base_avg,
    }
    with open(output_json, "w") as f:
        json.dump(summary_data, f, indent=4)

    print(f"✅ Real I/O Benchmark Results successfully saved to: {output_json}")


if __name__ == "__main__":
    run_real_evaluation()
