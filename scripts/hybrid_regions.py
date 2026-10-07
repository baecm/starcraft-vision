#!/usr/bin/env python3
"""
hybrid_regions.py
=================

Post-hoc combination of two methods' stored predictions, no training:
the primary region is the top box of a proposal detector, and the auxiliary
regions are a heatmap detector's (score >= tau_aux, not overlapping the
primary by >= delta of a viewport, at most k_max - 1 of them). It is compared
against the proposal detector's own top-k lists, which share that primary.

Metrics over all frames of the given replays, an unanswered frame scoring 0:
  regions  mean regions emitted
  IR       intersection ratio of the primary with the union of viewports
  OC3      fraction of spectators with >= delta of their viewport covered
  beta_n   slope of the region count on the number of modes (all frames)
  aux_new  share of auxiliary regions covering a new mode (multimodal frames)

Usage
-----
python scripts/hybrid_regions.py \
    --primary maskrcnn_win4_vanilla_f1_s123_v6@0.5 \
    --aux dc_full_b16_f1_s123_v6
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
    frame_slope,
    load_regions,
    load_replay_gt,
    mode_params,
    parse_run_spec,
    per_viewport_set,
)
from metrics.modes import box_from_center, box_mask, extract_modes


def region_lists(primary_boxes, aux_boxes, aux_scores, gt, args) -> dict:
    """The region lists compared on one frame, by variant name."""
    variants = {f"proposal_top{k}": primary_boxes[:k] for k in range(1, args.k_max + 1)}
    for tau in args.aux_taus:
        if len(primary_boxes):
            extra = [aux_boxes[i] for i in range(len(aux_boxes))
                     if aux_scores[i] >= tau
                     and _rect_overlap_ratio(aux_boxes[i], primary_boxes[0], gt.height, gt.width) < args.delta]
            variants[f"hybrid_aux>={tau:g}"] = np.array([primary_boxes[0]] + extra[: args.k_max - 1])
        else:
            variants[f"hybrid_aux>={tau:g}"] = primary_boxes
    return variants


def score_frame(st, boxes, viewports, union, mode_boxes, gt, delta) -> None:
    """Add one frame's IR, OC3 and auxiliary placement to the accumulator st."""
    h, w = gt.height, gt.width
    st["frames"] += 1
    st["regions"] += len(boxes)
    if len(boxes) == 0:
        return
    primary_mask = box_mask(boxes[0], h, w)
    st["IR"] += (primary_mask & union).sum() / primary_mask.sum()
    served = sum(max(_rect_overlap_ratio(b, vp, h, w) for b in boxes) >= delta for vp in viewports)
    st["OC3"] += served / len(viewports)
    if len(mode_boxes) >= 2:
        covered = set()
        for rank, b in enumerate(boxes):
            coverage = np.array([_rect_overlap_ratio(b, mb, h, w) for mb in mode_boxes])
            if rank > 0:
                best = int(np.argmax(coverage))
                st["aux"] += 1
                st["new"] += coverage[best] >= delta and best not in covered
            covered |= set(np.nonzero(coverage >= delta)[0].tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_replay_args(ap)
    add_prediction_args(ap)
    ap.add_argument("--primary", required=True, metavar="MODEL[@THRESHOLD]")
    ap.add_argument("--aux", required=True, metavar="MODEL[@THRESHOLD]")
    ap.add_argument("--aux-taus", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--delta", type=float, default=0.5)
    add_mode_args(ap)
    args = ap.parse_args()

    primary_model, primary_th = parse_run_spec(args.primary)
    aux_model, aux_th = parse_run_spec(args.aux)
    acc = defaultdict(lambda: defaultdict(float))
    n_modes, n_regions = defaultdict(list), defaultdict(list)  # per variant, per frame
    empty = (np.zeros((0, 4)), np.zeros(0))

    for replay in args.replays:
        gt = load_replay_gt(args.label_root, replay, args.label_method)
        h, w = gt.height, gt.width
        primary = load_regions(args.pred_root, primary_model, args.epoch, replay, args.label_method,
                               primary_th, gt.viewport_wh)
        aux = load_regions(args.pred_root, aux_model, args.epoch, replay, args.label_method,
                           aux_th, gt.viewport_wh)

        def centers_and_union(viewports):
            m = extract_modes(viewports, h, w, *mode_params(args))
            union = np.zeros((h, w), bool)
            for b in viewports:
                union |= box_mask(b, h, w)
            return m.centers, union

        modes = per_viewport_set(gt.viewports_by_frame, centers_and_union)
        for frame, viewports in gt.viewports_by_frame.items():
            centers, union = modes[frame]
            mode_boxes = [box_from_center(c, gt.viewport_hw, h, w) for c in centers]
            primary_boxes = primary.get(frame, empty)[0]
            aux_boxes, aux_scores = aux.get(frame, empty)
            for name, boxes in region_lists(primary_boxes, aux_boxes, aux_scores, gt, args).items():
                n_modes[name].append(len(centers))
                n_regions[name].append(len(boxes))
                score_frame(acc[name], boxes, viewports, union, mode_boxes, gt, args.delta)
        print(f"done {replay}", file=sys.stderr, flush=True)

    print("variant\tregions\tIR\tOC3\tbeta_n\taux_new")
    for name, st in acc.items():
        beta = frame_slope(np.array(n_modes[name], float), np.array(n_regions[name], float))
        fr = st["frames"]
        print(f"{name}\t{st['regions'] / fr:.2f}\t{st['IR'] / fr:.4f}\t{st['OC3'] / fr:.4f}\t"
              f"{beta:+.3f}\t{st['new'] / max(1, st['aux']):.3f}")


if __name__ == "__main__":
    main()
