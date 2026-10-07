#!/usr/bin/env python3
"""
aux_on_modes.py
===============

Do the auxiliary regions land on the spectator modes? For every frame with
two or more modes, each of a method's top-K regions (highest score first) is
assigned the mode it covers best, by the same overlap `attribute` uses for
the primary region in `src/metrics/modes.py`, and counted as:

  new   it covers (>= delta) a mode no higher-ranked region covers
  dup   its best mode is already covered by a higher-ranked region
  off   it covers no mode at delta

(the rule of analysis_common.classify_regions, inlined here because it also
needs the index of the mode covered) and, for `new` auxiliary regions,
whether the mode it covers is the highest-ranked mode still uncovered ("in
rank order"). Per frame it also reports how many distinct modes the regions
serve and what fraction of the minority modes (rank >= 2) any region covers.
A frame with no regions serves no mode. Frames are split by the support of
the second mode (m1: held by one spectator, m2: by two or more), as in
`split_by_second_mode.py`.

Unlike count_validity.py, only the top --k-max regions count.

Usage
-----
python scripts/aux_on_modes.py --replays 275 1725 3613 4520 4664 \
    --model dcn=dc_full_b16_f1_s123_v6 \
    --model mrcnn_th09=maskrcnn_win4_vanilla_f1_s123_v6@0.9
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

from analysis_common import (
    _rect_overlap_ratio,
    add_mode_args,
    add_prediction_args,
    add_replay_args,
    load_regions,
    load_replay_gt,
    mode_params,
    parse_model_spec,
    per_viewport_set,
)
from metrics.modes import box_from_center, extract_modes


def accumulate_frame(stats, groups, boxes, mode_boxes, gt, delta) -> None:
    """Classify one frame's regions and add them to stats[group] for each group."""
    n = len(mode_boxes)
    covered = set()
    for rank, box in enumerate(boxes):
        coverage = np.array([_rect_overlap_ratio(box, mb, gt.height, gt.width) for mb in mode_boxes])
        hit = set(np.nonzero(coverage >= delta)[0].tolist())
        if rank > 0:  # auxiliary region
            best = int(np.argmax(coverage))
            if coverage[best] < delta:
                kind = "off"
            elif best in covered:
                kind = "dup"
            else:
                kind = "new"
                first_uncovered = min(i for i in range(n) if i not in covered)
                for grp in groups:
                    stats[grp]["new_in_order"] += best == first_uncovered
            for grp in groups:
                stats[grp]["aux"] += 1
                stats[grp][kind] += 1
        covered |= hit
    for grp in groups:
        st = stats[grp]
        st["frames"] += 1
        st["regions"] += len(boxes)
        st["modes"] += n
        st["served"] += len(covered)
        st["minority"] += n - 1
        st["minority_served"] += len(covered - {0})
        st["top2_served"] += 1 in covered


def report(models, stats_by_model) -> None:
    print("method\tgroup\tframes\tregions\taux_per_frame\tnew\tdup\toff\tnew_in_order\t"
          "modes_served\tminority_recall\ttop2_recall")
    for spec in models:
        for grp in ("all", "m1", "m2"):
            st = stats_by_model[spec.name][grp]
            fr, aux = st["frames"], max(1.0, st["aux"])
            print("\t".join([spec.name, grp, f"{int(fr)}", f"{st['regions'] / fr:.2f}", f"{st['aux'] / fr:.2f}",
                             f"{st['new'] / aux:.3f}", f"{st['dup'] / aux:.3f}", f"{st['off'] / aux:.3f}",
                             f"{st['new_in_order'] / max(1.0, st['new']):.3f}",
                             f"{st['served'] / fr:.2f}", f"{st['minority_served'] / st['minority']:.3f}",
                             f"{st['top2_served'] / fr:.3f}"]), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_replay_args(ap)
    add_prediction_args(ap)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=MODEL_NAME[:EPOCH][@THRESHOLD]")
    ap.add_argument("--k-max", type=int, default=3, help="regions per frame that count")
    ap.add_argument("--delta", type=float, default=0.5)
    add_mode_args(ap)
    args = ap.parse_args()

    models = [parse_model_spec(s, args.epoch) for s in args.model]
    stats_by_model = {spec.name: defaultdict(lambda: defaultdict(float)) for spec in models}

    for replay in args.replays:
        gt = load_replay_gt(args.label_root, replay, args.label_method)

        def centers_and_support(viewports):
            m = extract_modes(viewports, gt.height, gt.width, *mode_params(args))
            return m.centers, m.support

        modes = per_viewport_set(gt.viewports_by_frame, centers_and_support)

        for spec in models:
            regions = load_regions(args.pred_root, spec.model, spec.epoch, replay,
                                   args.label_method, spec.threshold, gt.viewport_wh)
            for frame, (centers, support) in modes.items():
                if len(centers) < 2:
                    continue
                group = "m2" if support[1] >= 2 else "m1"
                boxes = regions.get(frame, (np.zeros((0, 4)), None))[0][: args.k_max]
                mode_boxes = [box_from_center(c, gt.viewport_hw, gt.height, gt.width) for c in centers]
                accumulate_frame(stats_by_model[spec.name], (group, "all"), boxes, mode_boxes, gt, args.delta)
        print(f"done {replay}", file=sys.stderr, flush=True)

    report(models, stats_by_model)


if __name__ == "__main__":
    main()
