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
  beta_n   slope of the region count on the number of modes
  aux_new  share of auxiliary regions covering a new mode (multimodal frames)

Usage
-----
python scripts/hybrid_regions.py \
    --primary maskrcnn_win4_vanilla_f1_s123_v6@0.5 \
    --aux dc_full_b16_f1_s123_v6
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

from evaluate import load_coco_gt, load_coco_preds  # noqa: E402
from metrics.modes import (  # noqa: E402
    _rect_overlap_ratio,
    box_from_center,
    box_mask,
    extract_modes,
    gt_boxes_by_frame,
    image_size,
    infer_region_size,
    predictions_from_dets,
)


def split_spec(spec: str):
    name, _, th = spec.partition("@")
    return name, (th or None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays", nargs="+", default=["275", "1725", "3613", "4520", "4664"])
    ap.add_argument("--label-root", default="/workspace/data/label/dst")
    ap.add_argument("--label-method", default="all_correct")
    ap.add_argument("--pred-root", default="/workspace/predictions")
    ap.add_argument("--primary", required=True, metavar="MODEL[@THRESHOLD]")
    ap.add_argument("--aux", required=True, metavar="MODEL[@THRESHOLD]")
    ap.add_argument("--aux-taus", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--delta", type=float, default=0.5)
    args = ap.parse_args()

    p_name, p_th = split_spec(args.primary)
    a_name, a_th = split_spec(args.aux)
    acc = defaultdict(lambda: defaultdict(float))
    xs, ys = defaultdict(list), defaultdict(list)

    for rep in args.replays:
        coco = load_coco_gt(args.label_root, rep, args.label_method)
        h, w = image_size(coco)
        gt = gt_boxes_by_frame(coco)
        size_hw = infer_region_size(gt)
        swh = (size_hw[1], size_hw[0])
        prim = predictions_from_dets(load_coco_preds(args.pred_root, p_name, args.epoch, rep, args.label_method, p_th), swh)
        aux = predictions_from_dets(load_coco_preds(args.pred_root, a_name, args.epoch, rep, args.label_method, a_th), swh)
        empty = (np.zeros((0, 4)), np.zeros(0))
        cache = {}
        for f, obs in gt.items():
            key = obs.tobytes()
            if key not in cache:
                m = extract_modes(obs, h, w, 4.0, 12.0, 0.35, 5)
                union = np.zeros((h, w), bool)
                for b in obs:
                    union |= box_mask(b, h, w)
                cache[key] = (m.centers, union)
            centers, union = cache[key]
            n = len(centers)
            mboxes = [box_from_center(c, size_hw, h, w) for c in centers]

            pb = prim.get(f, empty)[0]
            ab, ascore = aux.get(f, empty)
            variants = {f"proposal_top{k}": pb[:k] for k in range(1, args.k_max + 1)}
            for t in args.aux_taus:
                if len(pb):
                    extra = [ab[i] for i in range(len(ab))
                             if ascore[i] >= t and _rect_overlap_ratio(ab[i], pb[0], h, w) < args.delta]
                    variants[f"hybrid_aux>={t:g}"] = np.array([pb[0]] + extra[: args.k_max - 1])
                else:
                    variants[f"hybrid_aux>={t:g}"] = pb

            for name, boxes in variants.items():
                st = acc[name]
                st["frames"] += 1
                st["regions"] += len(boxes)
                xs[name].append(n)
                ys[name].append(len(boxes))
                if len(boxes) == 0:
                    continue
                pm = box_mask(boxes[0], h, w)
                st["IR"] += (pm & union).sum() / pm.sum()
                served = sum(max(_rect_overlap_ratio(b, o, h, w) for b in boxes) >= args.delta for o in obs)
                st["OC3"] += served / len(obs)
                if n >= 2:
                    covered = set()
                    for r, b in enumerate(boxes):
                        covs = np.array([_rect_overlap_ratio(b, mb, h, w) for mb in mboxes])
                        if r > 0:
                            best = int(np.argmax(covs))
                            st["aux"] += 1
                            st["new"] += covs[best] >= args.delta and best not in covered
                        covered |= set(np.nonzero(covs >= args.delta)[0].tolist())
        print(f"done {rep}", file=sys.stderr, flush=True)

    print("variant\tregions\tIR\tOC3\tbeta_n\taux_new")
    for name, st in acc.items():
        x, y = np.array(xs[name], float), np.array(ys[name], float)
        beta = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
        fr = st["frames"]
        print(f"{name}\t{st['regions'] / fr:.2f}\t{st['IR'] / fr:.4f}\t{st['OC3'] / fr:.4f}\t"
              f"{beta:+.3f}\t{st['new'] / max(1, st['aux']):.3f}")


if __name__ == "__main__":
    main()
