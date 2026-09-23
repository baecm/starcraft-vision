#!/usr/bin/env python3
"""
mode_disagreement.py
=====================

CLI for the attention-mode disagreement diagnostics in
`src/metrics/modes.py` (mode extraction, error attribution by mode, mode
flip rate, and reference multi-region metrics). See that module's docstring
for what each number means and how it relates to the project's existing
metrics (in particular why `mode_flip_rate` is not the same thing as
`evaluator.compute_m_cti`'s `jump_rate`).

GT and prediction files are resolved the same way `evaluate.py` resolves
them (`load_coco_gt` / `load_coco_preds`, re-exported by `estimate.py` for
backwards compatibility):
  GT   : {label-root}/{replay}.rep/{label-method}.json
  pred : {pred-root}/{model_name}/model_{epoch:03d}/{replay}.rep/{label-method}.json
so a run just needs replay ids plus the model folder names under
`predictions/` - no hand-built paths.

Usage
-----
python scripts/mode_disagreement.py \
    --replays 275 1725 3613 4520 4664 \
    --label-method all_correct \
    --model maskrcnn=maskrcnn_win4_vanilla_fold1_s123_20251201_072032 \
    --model maskrcnn_kbrs=maskrcnn_win4_kbrs_fold1_s123_kbrs025_base_score_20260203_055642 \
    --model centernet=centernet_vanilla_win1_fold1_s123_20260818_071116 \
    --epoch 30 \
    --outdir results/mode_disagreement/fold1

A model spec is NAME=MODEL_NAME, or NAME=MODEL_NAME:EPOCH to override
--epoch for just that model.

Per-frame records for the whole fold are written to
<outdir>/frames_<method>.csv (with a "replay" column) so any further
breakdown can be done without re-running.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

# Add repository root and src to python path (same convention as
# scripts/run_benchmark.py)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(root_dir, "src")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import numpy as np
import pandas as pd

from estimate import load_coco_gt, load_coco_preds
from metrics.modes import (
    analyse_method,
    by_margin,
    by_n_modes,
    by_quartile,
    by_replay,
    gt_boxes_by_frame,
    image_size,
    infer_region_size,
    predictions_from_dets,
    primary_track_m_cti,
    summarise,
)


def parse_model_spec(spec: str, default_epoch: int) -> Tuple[str, str, int]:
    name, sep, rest = spec.partition("=")
    if not sep:
        raise ValueError(f"--model spec must be NAME=MODEL_NAME[:EPOCH], got: {spec!r}")
    model_name, _, epoch_str = rest.partition(":")
    epoch = int(epoch_str) if epoch_str else default_epoch
    return name, model_name, epoch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays", type=str, nargs="+", required=True,
                    help="replay ids to pool into one fold-level result, e.g. 275 1725 3613 4520 4664")
    ap.add_argument("--label-root", default="/workspace/data/label/dst")
    ap.add_argument("--label-method", default="all_correct")
    ap.add_argument("--pred-root", default="/workspace/predictions")
    ap.add_argument("--model", action="append", required=True, metavar="NAME=MODEL_NAME[:EPOCH]",
                    help="prediction source under --pred-root, repeatable "
                         "(e.g. maskrcnn=<dir>, maskrcnn_kbrs=<dir>, centernet=<dir>)")
    ap.add_argument("--epoch", type=int, default=30,
                    help="default epoch for --model specs that omit :EPOCH")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=4.0,
                    help="Gaussian sigma (tiles) for smoothing the coverage map")
    ap.add_argument("--min-sep", type=float, default=12.0,
                    help="minimum separation D between modes (tiles)")
    ap.add_argument("--rel-threshold", type=float, default=0.35,
                    help="peak floor as a fraction of the frame's maximum")
    ap.add_argument("--max-modes", type=int, default=5)
    ap.add_argument("--straddle-floor", type=float, default=0.15,
                    help="second-best mode coverage above which a miss counts "
                         "as straddling two modes rather than off-mode")
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--jump-threshold", type=float, default=35.0,
                    help="displacement threshold (tiles) for the reference "
                         "M-CTI jump_rate, reported alongside mode_flip_rate")
    ap.add_argument("--region-size", type=int, nargs=2, default=None,
                    metavar=("H", "W"),
                    help="viewport size in tiles; inferred per-replay from the GT if omitted")
    ap.add_argument("--holdout", action="store_true",
                    help="also score a held-out observer. The analysis runs "
                         "once per observer with that one removed from the "
                         "ground truth, and HO<k>@delta reports whether the "
                         "regions the others produced still reach them. Costs "
                         "one pass per observer. Point --label-root at the "
                         "plain labels, not the _roci ones, or the augmented "
                         "viewports will be counted as people.")
    ap.add_argument("--num-observers", type=int, default=5,
                    help="observers per frame, used only to size --holdout")
    ap.add_argument("--raw-pred-size", action="store_true",
                    help="measure predictions at the width/height stored in the "
                         "prediction file instead of anchoring each box at its "
                         "top-left corner and forcing the GT viewport size. Only "
                         "for reproducing older numbers: a box clipped at the map "
                         "edge is stored narrower, which drags its computed centre "
                         "toward that edge and inflates corner statistics.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    models = [parse_model_spec(spec, args.epoch) for spec in args.model]
    print(f"replays: {args.replays}")
    print(f"models : {[(n, m, e) for n, m, e in models]}")
    print(f"pred box size: {'as stored in the prediction file' if args.raw_pred_size else 'GT viewport size, anchored at top-left'}\n")

    per_method_frames = {name: [] for name, _, _ in models}
    per_method_mcti = {name: [] for name, _, _ in models}

    for replay in args.replays:
        replay = str(replay)
        print(f"[Replay] {replay}")
        coco_gt = load_coco_gt(args.label_root, replay, args.label_method)
        height, width = image_size(coco_gt)
        gt_by_frame = gt_boxes_by_frame(coco_gt)
        if not gt_by_frame:
            print(f"  [!] no GT frames for replay {replay}, skipping")
            continue
        size_hw = tuple(args.region_size) if args.region_size else infer_region_size(gt_by_frame)
        # GT viewports all share one size, so it is also the size every
        # prediction denotes; box_slices/box_center take (x, y, w, h).
        pred_size_wh = None if args.raw_pred_size else (size_hw[1], size_hw[0])

        for name, model_name, epoch in models:
            dets_by_img = load_coco_preds(
                pred_root=args.pred_root, model_name=model_name, epoch=epoch,
                replay_id=replay, label_method=args.label_method,
            )
            pred_by_frame = predictions_from_dets(dets_by_img, pred_size_wh)
            if not pred_by_frame:
                print(f"  [!] {name}: no predictions for replay {replay}, skipping")
                continue

            # The baseline pass always runs, with every observer in the ground
            # truth, and it is the one the main figures come from. --holdout
            # adds a pass per observer on top, each with that one removed, and
            # those rows carry a `holdout` column so `summarise` can keep the
            # two apart. The flag is therefore additive: turning it on cannot
            # move a number that was reported without it.
            passes = [None]
            if args.holdout:
                passes += list(range(args.num_observers))

            replay_frames = []
            for ho in passes:
                tag = f"{replay}/{name}" if ho is None else f"{replay}/{name}/ho{ho}"
                df = analyse_method(
                    gt_by_frame, pred_by_frame, height, width,
                    size_hw=size_hw, sigma=args.sigma, min_sep=args.min_sep,
                    rel_threshold=args.rel_threshold, max_modes=args.max_modes,
                    delta=args.delta, straddle_floor=args.straddle_floor,
                    k_max=args.k_max, label=tag, holdout=ho,
                )
                df.insert(0, "replay", replay)
                replay_frames.append(df)

            df = pd.concat(replay_frames, ignore_index=True)
            per_method_frames[name].append(df)
            frames_sorted = sorted(df["frame"].unique())
            mcti = primary_track_m_cti(
                pred_by_frame, frames_sorted, args.jump_threshold
            )
            # The same trajectory under the other treatment of a declined
            # frame: hold the previous viewport rather than cut the run.
            mcti_ff = primary_track_m_cti(
                pred_by_frame, frames_sorted, args.jump_threshold,
                carry_forward=True,
            )
            mcti.update({f"{key}_ff": value for key, value in mcti_ff.items()})
            per_method_mcti[name].append(mcti)
        print()

    summaries = []
    for name, _, _ in models:
        dfs = per_method_frames[name]
        if not dfs:
            print(f"--- {name}: no data across the fold, skipping")
            continue
        print(f"--- {name}")
        df_all = pd.concat(dfs, ignore_index=True)
        df_all.to_csv(os.path.join(args.outdir, f"frames_{name}.csv"), index=False)

        row = {"method": name}
        row.update(summarise(df_all, args.delta, args.k_max))
        mcti_list = per_method_mcti[name]
        mcti_keys = (
            "m_cti", "jerk", "jump_rate", "velocity", "tracked_fraction",
            "m_cti_ff", "jerk_ff", "jump_rate_ff", "velocity_ff",
            "tracked_fraction_ff",
        )
        for key in mcti_keys:
            row[key] = float(np.mean([m[key] for m in mcti_list])) if mcti_list else float("nan")
        summaries.append(row)

        by_quartile(df_all, args.delta, args.k_max).to_csv(
            os.path.join(args.outdir, f"progression_{name}.csv"), index=False)
        by_margin(df_all, args.delta).to_csv(
            os.path.join(args.outdir, f"margin_{name}.csv"), index=False)
        by_n_modes(df_all, args.delta).to_csv(
            os.path.join(args.outdir, f"nmodes_{name}.csv"), index=False)
        by_replay(df_all, args.delta).to_csv(
            os.path.join(args.outdir, f"replay_{name}.csv"), index=False)

        print(by_replay(df_all, args.delta).to_string(index=False))
        print(by_margin(df_all, args.delta).to_string(index=False))
        print()

    if not summaries:
        print("[Info] No method produced any data across the fold.")
        return

    summary = pd.DataFrame(summaries)
    summary.to_csv(os.path.join(args.outdir, "summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print("=== summary")
    print(summary.to_string(index=False))
    print(f"\nwrote {args.outdir}/summary.csv and per-method CSVs")


if __name__ == "__main__":
    main()
