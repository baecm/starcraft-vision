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

and, for `new` auxiliary regions, whether the mode it covers is the
highest-ranked mode still uncovered ("in rank order"). Per frame it also
reports how many distinct modes the regions serve and what fraction of the
minority modes (rank >= 2) any region covers. A frame with no regions
serves no mode. Frames are split by the support of the second mode
(m1: held by one spectator, m2: by two or more), as in
`split_by_second_mode.py`.

Predictions are read and sized exactly as in `mode_disagreement.py`.

Usage
-----
python scripts/aux_on_modes.py --replays 275 1725 3613 4520 4664 \
    --model dcn=dc_full_b16_f1_s123_v6 \
    --model mrcnn_th09=maskrcnn_win4_vanilla_f1_s123_v6@0.9
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
    extract_modes,
    gt_boxes_by_frame,
    image_size,
    infer_region_size,
    predictions_from_dets,
)


def parse_model(spec: str):
    name, rest = spec.split("=", 1)
    th = None
    if "@" in rest:
        rest, th = rest.split("@", 1)
    return name, rest, th


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays", nargs="+", default=["275", "1725", "3613", "4520", "4664"])
    ap.add_argument("--label-root", default="/workspace/data/label/dst")
    ap.add_argument("--label-method", default="all_correct")
    ap.add_argument("--pred-root", default="/workspace/predictions")
    ap.add_argument("--model", action="append", required=True, metavar="NAME=MODEL_NAME[@THRESHOLD]")
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--min-sep", type=float, default=12.0)
    ap.add_argument("--rel-threshold", type=float, default=0.35)
    ap.add_argument("--max-modes", type=int, default=5)
    args = ap.parse_args()

    models = [parse_model(s) for s in args.model]
    stats = {name: defaultdict(lambda: defaultdict(float)) for name, _, _ in models}

    for rep in args.replays:
        coco = load_coco_gt(args.label_root, rep, args.label_method)
        h, w = image_size(coco)
        gt = gt_boxes_by_frame(coco)
        size_hw = infer_region_size(gt)
        cache = {}
        modes_by_frame = {}
        for f, boxes in gt.items():
            key = boxes.tobytes()
            if key not in cache:
                m = extract_modes(boxes, h, w, args.sigma, args.min_sep, args.rel_threshold, args.max_modes)
                cache[key] = (m.centers, m.support)
            modes_by_frame[f] = cache[key]

        for name, model_name, th in models:
            dets = load_coco_preds(args.pred_root, model_name, args.epoch, rep, args.label_method, th)
            preds = predictions_from_dets(dets, (size_hw[1], size_hw[0]))
            for f, (centers, support) in modes_by_frame.items():
                n = len(centers)
                if n < 2:
                    continue
                g = "m2" if support[1] >= 2 else "m1"
                boxes = preds.get(f, (np.zeros((0, 4)), None))[0][: args.k_max]
                mode_boxes = [box_from_center(c, size_hw, h, w) for c in centers]
                covered = set()
                for r, box in enumerate(boxes):
                    covs = np.array([_rect_overlap_ratio(box, mb, h, w) for mb in mode_boxes])
                    hit = set(np.nonzero(covs >= args.delta)[0].tolist())
                    if r > 0:
                        best = int(np.argmax(covs))
                        if covs[best] < args.delta:
                            kind = "off"
                        elif best in covered:
                            kind = "dup"
                        else:
                            kind = "new"
                            first_uncovered = min(i for i in range(n) if i not in covered)
                            for grp in (g, "all"):
                                stats[name][grp]["new_in_order"] += best == first_uncovered
                        for grp in (g, "all"):
                            stats[name][grp]["aux"] += 1
                            stats[name][grp][kind] += 1
                    covered |= hit
                for grp in (g, "all"):
                    st = stats[name][grp]
                    st["frames"] += 1
                    st["regions"] += len(boxes)
                    st["modes"] += n
                    st["served"] += len(covered)
                    st["minority"] += n - 1
                    st["minority_served"] += len(covered - {0})
                    st["top2_served"] += 1 in covered
        print(f"done {rep}", file=sys.stderr, flush=True)

    print("method\tgroup\tframes\tregions\taux_per_frame\tnew\tdup\toff\tnew_in_order\t"
          "modes_served\tminority_recall\ttop2_recall")
    for name, _, _ in models:
        for grp in ("all", "m1", "m2"):
            st = stats[name][grp]
            fr, aux = st["frames"], max(1.0, st["aux"])
            print("\t".join([name, grp, f"{int(fr)}", f"{st['regions'] / fr:.2f}", f"{st['aux'] / fr:.2f}",
                             f"{st['new'] / aux:.3f}", f"{st['dup'] / aux:.3f}", f"{st['off'] / aux:.3f}",
                             f"{st['new_in_order'] / max(1.0, st['new']):.3f}",
                             f"{st['served'] / fr:.2f}", f"{st['minority_served'] / st['minority']:.3f}",
                             f"{st['top2_served'] / fr:.3f}"]), flush=True)


if __name__ == "__main__":
    main()
