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

Usage
-----
python scripts/off_diagnosis.py --replays 275 1725 3613 4520 4664 \
    --model dcn=dc_full_b16_f1_s123_v6
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

MAX_N = 5
SCORE_BINS = [0.10, 0.12, 0.15, 0.20, 0.30, 1.01]
DIST_BINS = [0, 12, 24, 48, 1e9]
TAU_AUX = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
REL = [0.2, 0.3, 0.5]


def parse_model(spec: str):
    name, rest = spec.split("=", 1)
    th = None
    if "@" in rest:
        rest, th = rest.split("@", 1)
    return name, rest, th


def classify(boxes, mode_boxes, h, w, delta):
    covered, kinds = set(), []
    for box in boxes:
        covs = np.array([_rect_overlap_ratio(box, mb, h, w) for mb in mode_boxes])
        best = int(np.argmax(covs))
        if covs[best] < delta:
            kinds.append("off")
        elif best in covered:
            kinds.append("dup")
        else:
            kinds.append("new")
        covered |= set(np.nonzero(covs >= delta)[0].tolist())
    return kinds


def slope(per_n, key):
    pts = [(n, v[key] / v["frames"], v["frames"]) for n, v in per_n.items() if v["frames"]]
    ws = sum(f for _, _, f in pts)
    mx = sum(n * f for n, _, f in pts) / ws
    my = sum(y * f for _, y, f in pts) / ws
    num = sum(f * (n - mx) * (y - my) for n, y, f in pts)
    den = sum(f * (n - mx) ** 2 for n, _, f in pts)
    return num / den if den else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays", nargs="+", default=["275", "1725", "3613", "4520", "4664"])
    ap.add_argument("--label-root", default="/workspace/data/label/dst")
    ap.add_argument("--label-method", default="all_correct")
    ap.add_argument("--pred-root", default="/workspace/predictions")
    ap.add_argument("--model", action="append", required=True, metavar="NAME=MODEL_NAME[@THRESHOLD]")
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=24, help="frames either side for `nearby`")
    ap.add_argument("--k", type=int, default=3, help="regions scored by OC")
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--min-sep", type=float, default=12.0)
    ap.add_argument("--rel-threshold", type=float, default=0.35)
    ap.add_argument("--max-modes", type=int, default=5)
    args = ap.parse_args()

    models = [parse_model(s) for s in args.model]
    diag = {name: defaultdict(float) for name, _, _ in models}
    rules = [("tau", t) for t in TAU_AUX] + [("rel", r) for r in REL]
    filt = {name: {rule: defaultdict(lambda: defaultdict(float)) for rule in rules} for name, _, _ in models}

    for rep in args.replays:
        coco = load_coco_gt(args.label_root, rep, args.label_method)
        h, w = image_size(coco)
        gt = gt_boxes_by_frame(coco)
        size_hw = infer_region_size(gt)
        rh, rw = size_hw
        cache, modes = {}, {}
        for f, boxes in gt.items():
            key = boxes.tobytes()
            if key not in cache:
                m = extract_modes(boxes, h, w, args.sigma, args.min_sep, args.rel_threshold, args.max_modes)
                cache[key] = (np.asarray(m.centers, dtype=float).reshape(-1, 2),
                              [box_from_center(c, size_hw, h, w) for c in m.centers])
            modes[f] = cache[key]
        frames = sorted(modes)
        fset = set(frames)

        for name, model_name, th in models:
            dets = load_coco_preds(args.pred_root, model_name, args.epoch, rep, args.label_method, th)
            preds = predictions_from_dets(dets, (rw, rh))
            d = diag[name]
            for f in frames:
                centers, mode_boxes = modes[f]
                n = len(mode_boxes)
                if n < 1 or n > MAX_N:
                    continue
                boxes, scores = preds.get(f, (np.zeros((0, 4)), np.zeros(0)))
                scores = np.asarray(scores if scores is not None else np.ones(len(boxes)), dtype=float)
                kinds = classify(boxes, mode_boxes, h, w, args.delta)
                obs = gt[f]

                # 1. diagnosis of auxiliary regions
                for r in range(1, len(boxes)):
                    kind = kinds[r]
                    if kind == "dup":
                        d["dup"] += 1
                        continue
                    box, s = boxes[r], scores[r]
                    c = np.array([box[1] + box[3] / 2.0, box[0] + box[2] / 2.0])
                    dist = float(np.min(np.linalg.norm(centers - c, axis=1)))
                    spec = any(_rect_overlap_ratio(box, ob, h, w) >= args.delta for ob in obs)
                    near = False
                    for g in range(f - args.window, f + args.window + 1):
                        if g == f or g not in fset:
                            continue
                        if any(_rect_overlap_ratio(box, mb, h, w) >= args.delta for mb in modes[g][1]):
                            near = True
                            break
                    d[kind] += 1
                    d[f"{kind}_score_sum"] += s
                    d[f"{kind}_spec"] += spec
                    d[f"{kind}_near"] += near
                    d[f"{kind}_spec_or_near"] += spec or near
                    for i in range(len(SCORE_BINS) - 1):
                        if SCORE_BINS[i] <= s < SCORE_BINS[i + 1]:
                            d[f"{kind}_s{i}"] += 1
                    for i in range(len(DIST_BINS) - 1):
                        if DIST_BINS[i] <= dist < DIST_BINS[i + 1]:
                            d[f"{kind}_d{i}"] += 1

                # 2. filtering rules
                for rule in rules:
                    kind_r, val = rule
                    if len(boxes):
                        thr = val if kind_r == "tau" else val * scores[0]
                        keep = [0] + [r for r in range(1, len(boxes)) if scores[r] >= thr]
                    else:
                        keep = []
                    kb = boxes[keep] if keep else np.zeros((0, 4))
                    # kinds of kept regions are re-derived: dropping a region can
                    # turn a later duplicate into a new one
                    kk = classify(kb, mode_boxes, h, w, args.delta)
                    v = filt[name][rule][n]
                    v["frames"] += 1
                    v["n_hat"] += len(kb)
                    v["new"] += kk.count("new")
                    v["off"] += kk.count("off")
                    v["answered"] += len(kb) > 0
                    top = kb[: args.k]
                    served = sum(any(_rect_overlap_ratio(b, ob, h, w) >= args.delta for b in top) for ob in obs)
                    v["oc"] += served / max(1, len(obs))
            print(f"done {rep} {name}", file=sys.stderr, flush=True)

    for name, _, _ in models:
        d = diag[name]
        print(f"\n== {name}: auxiliary regions new={int(d['new'])} off={int(d['off'])} dup={int(d['dup'])}")
        print("kind\tmean_score\t" + "\t".join(f"s[{SCORE_BINS[i]:.2f},{SCORE_BINS[i+1]:.2f})" for i in range(len(SCORE_BINS) - 1))
              + "\t" + "\t".join(f"d[{DIST_BINS[i]:g},{DIST_BINS[i+1]:g})" for i in range(len(DIST_BINS) - 1))
              + "\tspectator\tnearby\tspec_or_near")
        for kind in ("new", "off"):
            c = max(1.0, d[kind])
            print("\t".join([kind, f"{d[f'{kind}_score_sum'] / c:.3f}"]
                            + [f"{d[f'{kind}_s{i}'] / c:.3f}" for i in range(len(SCORE_BINS) - 1)]
                            + [f"{d[f'{kind}_d{i}'] / c:.3f}" for i in range(len(DIST_BINS) - 1)]
                            + [f"{d[f'{kind}_spec'] / c:.3f}", f"{d[f'{kind}_near'] / c:.3f}",
                               f"{d[f'{kind}_spec_or_near'] / c:.3f}"]))
        print("\nrule\tregions\tnew\toff\tcoverage\tOC\tbeta_n\tbeta_new")
        for rule in rules:
            pn = filt[name][rule]
            fr = sum(v["frames"] for v in pn.values())
            tot = lambda k: sum(v[k] for v in pn.values()) / fr  # noqa: E731
            print("\t".join([f"{rule[0]}={rule[1]}", f"{tot('n_hat'):.3f}", f"{tot('new'):.3f}", f"{tot('off'):.3f}",
                             f"{tot('answered'):.3f}", f"{tot('oc'):.4f}",
                             f"{slope(pn, 'n_hat'):+.4f}", f"{slope(pn, 'new'):+.4f}"]), flush=True)


if __name__ == "__main__":
    main()
