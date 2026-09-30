#!/usr/bin/env python3
"""
switch_preparation.py
=====================

Does a ranked region set make the primary camera's cuts less abrupt?

The trajectory measurements in the paper say the primary region alternates
between competing modes at the same rate whichever formulation emits it: on the
margin-0 frames of fold 1 the proposal baseline and Director-CenterNet both
flip at about 0.073. So the multi-region output does not stop the camera from
moving. This script asks the different question a broadcast director would ask
about the same event:

    when the primary region jumps somewhere new, was that somewhere already
    on screen as an auxiliary region in the frame before?

A cut to a region the audience has been watching in a corner inset is a
different production event from a cut to a region they have not seen. The
paper's own framing - a primary plus a ranked set of auxiliary regions - is
worth something in exactly this sense, and nothing in the accuracy or count
metrics captures it.

Definitions, all spatial, so no correspondence between one frame's extracted
modes and the next one's is needed:

    switch at t     overlap(primary_t, primary_{t+1}) < delta.  The camera has
                    moved somewhere that the previous primary did not already
                    cover.
    prepared        some auxiliary region k >= 2 at frame t overlaps
                    primary_{t+1} by at least delta.  The destination was on
                    screen before the cut.

Both are computed identically for every method, on its own top-K, which is the
protocol the paper uses everywhere else: the baseline emits a ranked list too,
and its k >= 2 boxes are its auxiliary regions.

The region count is a confound, as it is throughout this comparison - a method
emitting six regions has more chances to have covered the destination than one
emitting two. Run the baseline at the threshold that matches the proposed
method's region count as well as at its default, exactly as
Section "Region Count Under a Matched Budget" does, and read the two together.

Usage
-----
  python3 scripts/switch_preparation.py \
      --replays 275 1725 3613 4520 4664 \
      --model director=dc_full_b16_f1_s456_v6 \
      --model maskrcnn=maskrcnn_win4_vanilla_f1_s456_v6@0.5 \
      --model maskrcnn_matched=maskrcnn_win4_vanilla_f1_s456_v6@0.9 \
      --epoch 30 --out /workspace/results/switch_preparation_s456.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(root_dir, "src")
for p in (root_dir, src_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluate import load_coco_gt, load_coco_preds  # noqa: E402
from metrics.modes import (  # noqa: E402
    _rect_overlap_ratio,
    box_center,
    gt_boxes_by_frame,
    image_size,
    infer_region_size,
    predictions_from_dets,
)


def parse_spec(spec: str, default_epoch: int) -> Tuple[str, str, int, float | None]:
    name, _, rest = spec.partition("=")
    if not rest:
        raise ValueError(f"--model wants NAME=MODEL[:EPOCH][@THRESHOLD], got {spec!r}")
    model, _, thr = rest.partition("@")
    model, _, ep = model.partition(":")
    return name, model, int(ep) if ep else default_epoch, float(thr) if thr else None


def load_mode_counts(path: str) -> Dict[Tuple[str, int], int]:
    """(replay, frame) -> |M_t|, from any frames_*.csv mode_disagreement wrote.

    The mode count is a property of the ground truth, so every method's CSV
    carries the same one and it does not matter which is passed.
    """
    df = pd.read_csv(path, usecols=["replay", "frame", "n_modes"])
    return {(str(r), int(f)): int(n)
            for r, f, n in zip(df["replay"], df["frame"], df["n_modes"])}


def analyse_replay(pred_by_frame: Dict[int, Tuple[np.ndarray, np.ndarray]],
                   frames: List[int], height: int, width: int,
                   delta: float, k: int) -> List[dict]:
    """One row per consecutive frame pair on which the primary is defined."""
    rows = []
    for a, b in zip(frames, frames[1:]):
        if b != a + 1:
            continue  # only genuinely consecutive frames
        boxes_a, _ = pred_by_frame.get(a, (np.empty((0, 4)), np.empty(0)))
        boxes_b, _ = pred_by_frame.get(b, (np.empty((0, 4)), np.empty(0)))
        if len(boxes_a) == 0 or len(boxes_b) == 0:
            continue

        prim_a, prim_b = boxes_a[0], boxes_b[0]
        held = _rect_overlap_ratio(prim_a, prim_b, height, width)
        switched = held < delta

        # Auxiliary regions the method was showing at frame a, capped at its
        # top-k the way every other metric in the paper caps it.
        aux_a = boxes_a[1:k]
        prepared_cov = max(
            (_rect_overlap_ratio(x, prim_b, height, width) for x in aux_a),
            default=0.0,
        )
        rows.append({
            "frame": a,
            "n_pred": len(boxes_a),
            "primary_hold": held,
            "switched": bool(switched),
            "prepared_cov": prepared_cov,
            "prepared": bool(switched and prepared_cov >= delta),
            "jump_tiles": float(np.linalg.norm(box_center(prim_a) - box_center(prim_b))),
        })
    return rows


def summarise(df: pd.DataFrame, name: str, scope: str = "all") -> dict:
    sw = df[df["switched"]]
    return {
        "method": name,
        "scope": scope,
        "pairs": len(df),
        "switch_rate": df["switched"].mean() if len(df) else np.nan,
        "switches": len(sw),
        # The headline: of the cuts this method makes, how many went somewhere
        # it was already showing.
        "prepared_rate": sw["prepared"].mean() if len(sw) else np.nan,
        "mean_prepared_cov_on_switch": sw["prepared_cov"].mean() if len(sw) else np.nan,
        "mean_jump_tiles": sw["jump_tiles"].mean() if len(sw) else np.nan,
        "mean_n_pred": df["n_pred"].mean() if len(df) else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays", nargs="+", required=True)
    ap.add_argument("--model", action="append", required=True,
                    metavar="NAME=MODEL[:EPOCH][@THRESHOLD]")
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--label-root", default="/workspace/data/label/dst")
    ap.add_argument("--label-method", default="all_correct")
    ap.add_argument("--pred-root", default="/workspace/predictions")
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=3, help="regions scored per frame")
    ap.add_argument("--modes-csv", default=None,
                    help="a frames_*.csv from mode_disagreement.py. With it the "
                         "summary is broken down by |M_t|, since a unimodal "
                         "frame has no second region for an auxiliary to have "
                         "been holding and dilutes the rate.")
    ap.add_argument("--out", default=None, help="write the summary to this CSV")
    args = ap.parse_args()

    modes_by_frame = load_mode_counts(args.modes_csv) if args.modes_csv else {}
    models = [parse_spec(s, args.epoch) for s in args.model]
    per_method: Dict[str, List[pd.DataFrame]] = {n: [] for n, _, _, _ in models}

    for replay in args.replays:
        coco = load_coco_gt(args.label_root, replay, args.label_method)
        gt = gt_boxes_by_frame(coco)
        height, width = image_size(coco)
        size_hw = infer_region_size(gt)
        print(f"[Replay] {replay}: {len(gt)} gt frames, viewport {size_hw}", flush=True)

        for name, model, epoch, thr in models:
            dets = load_coco_preds(pred_root=args.pred_root, model_name=model,
                                   epoch=epoch, replay_id=replay,
                                   label_method=args.label_method,
                                   score_threshold=thr)
            if not dets:
                print(f"  [!] {name}: no predictions, skipping", flush=True)
                continue
            pred_by_frame = predictions_from_dets(dets, (size_hw[1], size_hw[0]))
            frames = sorted(f for f in pred_by_frame if f in gt)
            rows = analyse_replay(pred_by_frame, frames, height, width, args.delta, args.k)
            if rows:
                d = pd.DataFrame(rows)
                d.insert(0, "replay", replay)
                if modes_by_frame:
                    d["n_modes"] = [modes_by_frame.get((str(replay), int(f)), -1)
                                    for f in d["frame"]]
                per_method[name].append(d)
            print(f"  {name}: {len(rows)} consecutive pairs", flush=True)

    out = []
    for name, _, _, _ in models:
        if not per_method[name]:
            continue
        df = pd.concat(per_method[name], ignore_index=True)
        out.append(summarise(df, name))
        if "n_modes" in df.columns:
            known = df[df["n_modes"] > 0]
            missing = len(df) - len(known)
            if missing:
                print(f"  [!] {name}: {missing} pairs absent from --modes-csv, "
                      f"excluded from the per-|M_t| rows", flush=True)
            out.append(summarise(known[known["n_modes"] >= 2], name, "n>=2"))
            for n in (1, 2, 3):
                sel = known[known["n_modes"] == n] if n < 3 else known[known["n_modes"] >= 3]
                if len(sel):
                    out.append(summarise(sel, name, f"n={n}" if n < 3 else "n>=3"))

    summary = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nswitch_rate    : fraction of consecutive pairs on which the primary "
          "moved off what it was covering")
    print("prepared_rate  : of those, the fraction whose destination an auxiliary "
          "region was already covering")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        summary.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
