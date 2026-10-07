#!/usr/bin/env python3
"""
count_validity.py
=================

Is the count response beta_n made of useful regions? beta_n is the weighted
slope of E[n_hat | n] on the number of modes n, and it says nothing about
where the regions go. Here every emitted region of a frame (highest score
first, no cap, as beta_n counts them) is classified against the frame's
ranked modes (analysis_common.classify_regions):

  new   it covers (>= delta) a mode no higher-ranked region covers
  dup   its best mode is already covered by a higher-ranked region
  off   it covers no mode at delta

so that n_hat = new + dup + off on every frame and, the slope being linear
in the per-n means, beta_n = beta_new + beta_dup + beta_off exactly.
beta_new is the slope of the number of distinct modes served.

Chance baselines keep each frame's own region count and replace the
positions:

  uniform  top-left corners drawn uniformly over the map
  prior    positions drawn from the method's own regions elsewhere in the
           same replay, which keeps where the method tends to look but
           breaks the link to this frame

A beta_new above chance means the added regions find the added modes more
often than the same number of regions placed without looking at the frame.

All frames with 1..5 modes enter, as in `budget_allocation.py`. A declined
frame has n_hat = 0.

Reported in: thesis tab:mrvp:countsplit, and the ESWA paper.

Usage
-----
python scripts/count_validity.py --replays 275 1725 3613 4520 4664 \
    --model dcn=dc_full_b16_f1_s123_v6 \
    --model mrcnn_th09=maskrcnn_win4_vanilla_f1_s123_v6@0.9
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

from analysis_common import (
    MAX_MODES_FOR_SLOPE,
    add_mode_args,
    add_prediction_args,
    add_replay_args,
    classify_regions,
    load_regions,
    load_replay_gt,
    mode_params,
    parse_model_spec,
    per_viewport_set,
    weighted_slope,
)
from metrics.modes import box_from_center, extract_modes

# Per-n accumulators: n_hat = new + dup + off, and the chance baselines' new.
PARTS = ("n_hat", "new", "dup", "off", "new_uniform", "new_prior")


def count_new(boxes, mode_boxes, gt, delta) -> int:
    return classify_regions(boxes, mode_boxes, gt.height, gt.width, delta).count("new")


def uniform_regions(rng, k: int, gt) -> np.ndarray:
    """k viewport-sized regions with top-left corners uniform over the map."""
    rh, rw = gt.viewport_hw
    boxes = np.zeros((k, 4))
    if k:
        boxes[:, 0] = rng.uniform(0, max(0, gt.width - rw), k)
        boxes[:, 1] = rng.uniform(0, max(0, gt.height - rh), k)
        boxes[:, 2], boxes[:, 3] = rw, rh
    return boxes


def prior_regions(rng, k: int, pool: np.ndarray) -> np.ndarray:
    """k regions drawn (with replacement) from the method's regions in this replay."""
    return pool[rng.integers(0, len(pool), k)] if k and len(pool) else np.zeros((0, 4))


def accumulate_replay(per_n, gt, regions, mode_boxes_by_frame, delta, rng) -> None:
    """Add one model's frames of one replay to per_n[n][part].

    The two chance draws use the same rng in a fixed order (uniform, then
    prior, frame by frame), so a run is reproducible from --seed.
    """
    all_regions = (np.concatenate([b for b, _ in regions.values() if len(b)])
                   if regions else np.zeros((0, 4)))
    for frame, mode_boxes in mode_boxes_by_frame.items():
        n = len(mode_boxes)
        if n < 1 or n > MAX_MODES_FOR_SLOPE:
            continue
        boxes = regions.get(frame, (np.zeros((0, 4)), None))[0]
        k = len(boxes)
        kinds = classify_regions(boxes, mode_boxes, gt.height, gt.width, delta)
        uniform = uniform_regions(rng, k, gt)
        prior = prior_regions(rng, k, all_regions)

        acc = per_n[n]
        acc["frames"] += 1
        acc["n_hat"] += k
        acc["new"] += kinds.count("new")
        acc["dup"] += kinds.count("dup")
        acc["off"] += kinds.count("off")
        acc["new_uniform"] += count_new(uniform, mode_boxes, gt, delta)
        acc["new_prior"] += count_new(prior, mode_boxes, gt, delta)


def part_slope(per_n, part: str) -> float:
    """beta of one part: the weighted slope of its per-n mean."""
    return weighted_slope([(n, acc[part] / acc["frames"], acc["frames"])
                           for n, acc in per_n.items() if acc["frames"]])


def report(models, per_n_by_model) -> None:
    print("method\tframes\t" + "\t".join(f"mean_{p}" for p in PARTS) + "\t"
          + "\t".join(f"beta_{p}" for p in PARTS))
    for spec in models:
        per_n = per_n_by_model[spec.name]
        frames = sum(acc["frames"] for acc in per_n.values())
        means = [sum(acc[p] for acc in per_n.values()) / frames for p in PARTS]
        betas = [part_slope(per_n, p) for p in PARTS]
        print("\t".join([spec.name, f"{int(frames)}"] + [f"{m:.3f}" for m in means]
                        + [f"{b:+.4f}" for b in betas]), flush=True)

    print("\nper n: method n frames " + " ".join(PARTS))
    for spec in models:
        per_n = per_n_by_model[spec.name]
        for n in sorted(per_n):
            acc = per_n[n]
            print(spec.name, n, int(acc["frames"]), " ".join(f"{acc[p] / acc['frames']:.3f}" for p in PARTS))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_replay_args(ap)
    add_prediction_args(ap)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=MODEL_NAME[:EPOCH][@THRESHOLD]")
    ap.add_argument("--delta", type=float, default=0.5)
    add_mode_args(ap)
    ap.add_argument("--seed", type=int, default=0, help="seed of the chance baselines")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    models = [parse_model_spec(s, args.epoch) for s in args.model]
    per_n_by_model = {spec.name: defaultdict(lambda: defaultdict(float)) for spec in models}

    for replay in args.replays:
        gt = load_replay_gt(args.label_root, replay, args.label_method)
        mode_boxes_by_frame = per_viewport_set(
            gt.viewports_by_frame,
            lambda viewports: [box_from_center(c, gt.viewport_hw, gt.height, gt.width)
                               for c in extract_modes(viewports, gt.height, gt.width, *mode_params(args)).centers])
        for spec in models:
            regions = load_regions(args.pred_root, spec.model, spec.epoch, replay,
                                   args.label_method, spec.threshold, gt.viewport_wh)
            accumulate_replay(per_n_by_model[spec.name], gt, regions, mode_boxes_by_frame, args.delta, rng)
            print(f"done {replay} {spec.name}", file=sys.stderr, flush=True)

    report(models, per_n_by_model)


if __name__ == "__main__":
    main()
