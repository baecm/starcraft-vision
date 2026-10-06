#!/usr/bin/env python3
"""
count_validity.py
=================

Is the count response beta_n made of useful regions? beta_n is the weighted
slope of E[n_hat | n] on the number of modes n, and it says nothing about
where the regions go. Here every emitted region of a frame (highest score
first, no cap, as beta_n counts them) is classified against the frame's
ranked modes, with the overlap `aux_on_modes.py` uses:

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

Usage
-----
python scripts/count_validity.py --replays 275 1725 3613 4520 4664 \
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

MAX_N = 5
PARTS = ("n_hat", "new", "dup", "off", "new_uniform", "new_prior")


def parse_model(spec: str):
    name, rest = spec.split("=", 1)
    th = None
    if "@" in rest:
        rest, th = rest.split("@", 1)
    return name, rest, th


def classify(boxes, mode_boxes, h, w, delta):
    """(new, dup, off) counts for regions in score order."""
    covered, new, dup, off = set(), 0, 0, 0
    for box in boxes:
        covs = np.array([_rect_overlap_ratio(box, mb, h, w) for mb in mode_boxes])
        best = int(np.argmax(covs))
        if covs[best] < delta:
            off += 1
        elif best in covered:
            dup += 1
        else:
            new += 1
        covered |= set(np.nonzero(covs >= delta)[0].tolist())
    return new, dup, off


def slope(per_n: dict, key: str) -> float:
    pts = [(n, v[key] / v["frames"], v["frames"]) for n, v in per_n.items() if v["frames"]]
    wsum = sum(f for _, _, f in pts)
    mx = sum(n * f for n, _, f in pts) / wsum
    my = sum(y * f for _, y, f in pts) / wsum
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
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--min-sep", type=float, default=12.0)
    ap.add_argument("--rel-threshold", type=float, default=0.35)
    ap.add_argument("--max-modes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    models = [parse_model(s) for s in args.model]
    per_n = {name: defaultdict(lambda: defaultdict(float)) for name, _, _ in models}

    for rep in args.replays:
        coco = load_coco_gt(args.label_root, rep, args.label_method)
        h, w = image_size(coco)
        gt = gt_boxes_by_frame(coco)
        size_hw = infer_region_size(gt)
        rh, rw = size_hw
        cache, modes_by_frame = {}, {}
        for f, boxes in gt.items():
            key = boxes.tobytes()
            if key not in cache:
                m = extract_modes(boxes, h, w, args.sigma, args.min_sep, args.rel_threshold, args.max_modes)
                cache[key] = [box_from_center(c, size_hw, h, w) for c in m.centers]
            modes_by_frame[f] = cache[key]

        for name, model_name, th in models:
            dets = load_coco_preds(args.pred_root, model_name, args.epoch, rep, args.label_method, th)
            preds = predictions_from_dets(dets, (rw, rh))
            pool = np.concatenate([b for b, _ in preds.values() if len(b)]) if preds else np.zeros((0, 4))
            for f, mode_boxes in modes_by_frame.items():
                n = len(mode_boxes)
                if n < 1 or n > MAX_N:
                    continue
                boxes = preds.get(f, (np.zeros((0, 4)), None))[0]
                k = len(boxes)
                new, dup, off = classify(boxes, mode_boxes, h, w, args.delta)
                uni = np.zeros((k, 4))
                if k:
                    uni[:, 0] = rng.uniform(0, max(0, w - rw), k)
                    uni[:, 1] = rng.uniform(0, max(0, h - rh), k)
                    uni[:, 2], uni[:, 3] = rw, rh
                pri = pool[rng.integers(0, len(pool), k)] if k and len(pool) else np.zeros((0, 4))
                v = per_n[name][n]
                v["frames"] += 1
                v["n_hat"] += k
                v["new"] += new
                v["dup"] += dup
                v["off"] += off
                v["new_uniform"] += classify(uni, mode_boxes, h, w, args.delta)[0]
                v["new_prior"] += classify(pri, mode_boxes, h, w, args.delta)[0]
            print(f"done {rep} {name}", file=sys.stderr, flush=True)

    print("method\tframes\t" + "\t".join(f"mean_{p}" for p in PARTS) + "\t"
          + "\t".join(f"beta_{p}" for p in PARTS))
    for name, _, _ in models:
        d = per_n[name]
        fr = sum(v["frames"] for v in d.values())
        means = [sum(v[p] for v in d.values()) / fr for p in PARTS]
        betas = [slope(d, p) for p in PARTS]
        print("\t".join([name, f"{int(fr)}"] + [f"{m:.3f}" for m in means] + [f"{b:+.4f}" for b in betas]),
              flush=True)
    print("\nper n: method n frames " + " ".join(PARTS))
    for name, _, _ in models:
        for n in sorted(per_n[name]):
            v = per_n[name][n]
            print(name, n, int(v["frames"]), " ".join(f"{v[p] / v['frames']:.3f}" for p in PARTS))


if __name__ == "__main__":
    main()
