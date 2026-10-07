#!/usr/bin/env python3
"""
off_diagnosis.py
================

What are the auxiliary regions that cover no mode, and can they be filtered
without retraining? Two parts, on stored predictions only.

1. Diagnosis. Every auxiliary region (rank >= 2 in score order) is classified
   new / dup / off against the frame's ranked modes, as in
   `count_validity.py`. For new and off regions it records

     score      the region's confidence
     dist       tiles from the region center to the nearest mode center
     spectator  whether it covers (>= delta) at least one spectator's own
                viewport, i.e. a place someone watched that did not become
                a mode (the extraction floor or the separation removed it)
     nearby     whether it covers a mode of a frame within +-W frames, i.e.
                a mode that was there a moment earlier or later

2. Filtering. The primary region is kept as decoded; auxiliary regions are
   kept only if their score clears tau_aux, or a fraction r of the primary's
   score. For each rule it reports regions, new and off per frame, beta_n,
   beta_new, OC3@delta and coverage.

All frames with one to five modes; a declined frame has no regions.

Reported in: thesis sec:mrvp:budget (off regions), ESWA Section 5.6.

Usage
-----
python scripts/off_diagnosis.py --replays 275 1725 3613 4520 4664 \
    --model dcn=dc_full_b16_f1_s123_v6
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

from analysis_common import (
    MAX_MODES_FOR_SLOPE,
    _rect_overlap_ratio,
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

# Histogram bins of part 1: region score, and distance (tiles) to the nearest mode.
SCORE_BINS = [0.10, 0.12, 0.15, 0.20, 0.30, 1.01]
DIST_BINS = [0, 12, 24, 48, 1e9]
# Filtering rules of part 2: an absolute score floor tau_aux, or a fraction of
# the primary region's score.
TAU_AUX = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
REL = [0.2, 0.3, 0.5]
RULES = [("tau", t) for t in TAU_AUX] + [("rel", r) for r in REL]


def covers_any(box, others, gt, delta) -> bool:
    """Whether >= delta of `box`'s own area lies inside one of `others`."""
    return any(_rect_overlap_ratio(box, o, gt.height, gt.width) >= delta for o in others)


def diagnose_aux_regions(diag, gt, frame, boxes, scores, kinds, modes, frame_set, args) -> None:
    """Part 1: add the auxiliary regions of one frame to the `diag` counters."""
    centers, _ = modes[frame]
    viewports = gt.viewports_by_frame[frame]
    for r in range(1, len(boxes)):
        kind = kinds[r]
        if kind == "dup":
            diag["dup"] += 1
            continue
        box, score = boxes[r], scores[r]
        center_rc = np.array([box[1] + box[3] / 2.0, box[0] + box[2] / 2.0])
        dist = float(np.min(np.linalg.norm(centers - center_rc, axis=1)))
        on_spectator = covers_any(box, viewports, gt, args.delta)
        nearby = False
        for g in range(frame - args.window, frame + args.window + 1):
            if g == frame or g not in frame_set:
                continue
            if covers_any(box, modes[g][1], gt, args.delta):
                nearby = True
                break
        diag[kind] += 1
        diag[f"{kind}_score_sum"] += score
        diag[f"{kind}_spec"] += on_spectator
        diag[f"{kind}_near"] += nearby
        diag[f"{kind}_spec_or_near"] += on_spectator or nearby
        for i in range(len(SCORE_BINS) - 1):
            if SCORE_BINS[i] <= score < SCORE_BINS[i + 1]:
                diag[f"{kind}_s{i}"] += 1
        for i in range(len(DIST_BINS) - 1):
            if DIST_BINS[i] <= dist < DIST_BINS[i + 1]:
                diag[f"{kind}_d{i}"] += 1


def apply_filter_rules(filtered, gt, frame, boxes, scores, mode_boxes, args) -> None:
    """Part 2: score one frame under every filtering rule."""
    n = len(mode_boxes)
    viewports = gt.viewports_by_frame[frame]
    for rule in RULES:
        rule_kind, value = rule
        if len(boxes):
            floor = value if rule_kind == "tau" else value * scores[0]
            keep = [0] + [r for r in range(1, len(boxes)) if scores[r] >= floor]
        else:
            keep = []
        kept = boxes[keep] if keep else np.zeros((0, 4))
        # kinds of kept regions are re-derived: dropping a region can turn a
        # later duplicate into a new one
        kinds = classify_regions(kept, mode_boxes, gt.height, gt.width, args.delta)
        acc = filtered[rule][n]
        acc["frames"] += 1
        acc["n_hat"] += len(kept)
        acc["new"] += kinds.count("new")
        acc["off"] += kinds.count("off")
        acc["answered"] += len(kept) > 0
        top_k = kept[: args.k]
        # a spectator is served when some region has >= delta of its own area
        # inside that spectator's viewport
        served = sum(any(_rect_overlap_ratio(b, vp, gt.height, gt.width) >= args.delta for b in top_k)
                     for vp in viewports)
        acc["oc"] += served / max(1, len(viewports))


def report(models, diag_by_model, filtered_by_model) -> None:
    for spec in models:
        diag = diag_by_model[spec.name]
        print(f"\n== {spec.name}: auxiliary regions new={int(diag['new'])} off={int(diag['off'])} dup={int(diag['dup'])}")
        print("kind\tmean_score\t" + "\t".join(f"s[{SCORE_BINS[i]:.2f},{SCORE_BINS[i+1]:.2f})" for i in range(len(SCORE_BINS) - 1))
              + "\t" + "\t".join(f"d[{DIST_BINS[i]:g},{DIST_BINS[i+1]:g})" for i in range(len(DIST_BINS) - 1))
              + "\tspectator\tnearby\tspec_or_near")
        for kind in ("new", "off"):
            count = max(1.0, diag[kind])
            print("\t".join([kind, f"{diag[f'{kind}_score_sum'] / count:.3f}"]
                            + [f"{diag[f'{kind}_s{i}'] / count:.3f}" for i in range(len(SCORE_BINS) - 1)]
                            + [f"{diag[f'{kind}_d{i}'] / count:.3f}" for i in range(len(DIST_BINS) - 1)]
                            + [f"{diag[f'{kind}_spec'] / count:.3f}", f"{diag[f'{kind}_near'] / count:.3f}",
                               f"{diag[f'{kind}_spec_or_near'] / count:.3f}"]))

        print("\nrule\tregions\tnew\toff\tcoverage\tOC\tbeta_n\tbeta_new")
        for rule in RULES:
            per_n = filtered_by_model[spec.name][rule]
            frames = sum(acc["frames"] for acc in per_n.values())

            def mean(key):
                return sum(acc[key] for acc in per_n.values()) / frames

            def slope(key):
                return weighted_slope([(n, acc[key] / acc["frames"], acc["frames"])
                                       for n, acc in per_n.items() if acc["frames"]])

            print("\t".join([f"{rule[0]}={rule[1]}", f"{mean('n_hat'):.3f}", f"{mean('new'):.3f}", f"{mean('off'):.3f}",
                             f"{mean('answered'):.3f}", f"{mean('oc'):.4f}",
                             f"{slope('n_hat'):+.4f}", f"{slope('new'):+.4f}"]), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_replay_args(ap)
    add_prediction_args(ap)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=MODEL_NAME[:EPOCH][@THRESHOLD]")
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=24, help="frames either side for `nearby`")
    ap.add_argument("--k", type=int, default=3, help="regions scored by OC")
    add_mode_args(ap)
    args = ap.parse_args()

    models = [parse_model_spec(s, args.epoch) for s in args.model]
    diag_by_model = {spec.name: defaultdict(float) for spec in models}
    filtered_by_model = {spec.name: {rule: defaultdict(lambda: defaultdict(float)) for rule in RULES}
                         for spec in models}

    for replay in args.replays:
        gt = load_replay_gt(args.label_root, replay, args.label_method)

        def centers_and_boxes(viewports):
            m = extract_modes(viewports, gt.height, gt.width, *mode_params(args))
            return (np.asarray(m.centers, dtype=float).reshape(-1, 2),
                    [box_from_center(c, gt.viewport_hw, gt.height, gt.width) for c in m.centers])

        modes = per_viewport_set(gt.viewports_by_frame, centers_and_boxes)
        frames = sorted(modes)
        frame_set = set(frames)

        for spec in models:
            regions = load_regions(args.pred_root, spec.model, spec.epoch, replay,
                                   args.label_method, spec.threshold, gt.viewport_wh)
            for frame in frames:
                mode_boxes = modes[frame][1]
                if len(mode_boxes) < 1 or len(mode_boxes) > MAX_MODES_FOR_SLOPE:
                    continue
                boxes, scores = regions.get(frame, (np.zeros((0, 4)), np.zeros(0)))
                scores = np.asarray(scores if scores is not None else np.ones(len(boxes)), dtype=float)
                kinds = classify_regions(boxes, mode_boxes, gt.height, gt.width, args.delta)
                diagnose_aux_regions(diag_by_model[spec.name], gt, frame, boxes, scores, kinds,
                                     modes, frame_set, args)
                apply_filter_rules(filtered_by_model[spec.name], gt, frame, boxes, scores, mode_boxes, args)
            print(f"done {replay} {spec.name}", file=sys.stderr, flush=True)

    report(models, diag_by_model, filtered_by_model)


if __name__ == "__main__":
    main()
