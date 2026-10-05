#!/usr/bin/env python3
"""
mode_sensitivity.py
===================

Sensitivity of the spectator mode structure to the extraction parameters
of `src/metrics/modes.extract_modes` (Gaussian width sigma, relative floor
theta, minimum separation D). Ground truth only: no model is involved.

The extraction is the same as `extract_modes`, reorganized so that one
frame's coverage is smoothed once per sigma and frames whose viewports
repeat are computed once.

For each setting it reports, over all frames of the given replays:
  meanM     mean number of ranked modes
  multi     fraction of frames with two or more modes
  multi_s2  fraction whose second mode is held by at least two spectators
  tie       fraction whose top two modes have equal support
  tie22     fraction tied with support of at least two each
  M_q1..q4  mean mode count per progression quarter, and the peak quarter

Usage
-----
python scripts/mode_sensitivity.py \
    --replays 275 1725 3613 4520 4664 \
    --label-root /workspace/data/label/dst --label-method all_correct
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

from evaluate import load_coco_gt  # noqa: E402
from metrics.modes import box_mask, gt_boxes_by_frame, image_size  # noqa: E402


def ranked_support(masks, smoothed, theta, sep, max_modes):
    """Supports of the ranked modes, as extract_modes computes them."""
    peak = smoothed >= maximum_filter(smoothed, size=3, mode="constant")
    peak &= smoothed >= theta * smoothed.max()
    rows, cols = np.nonzero(peak)
    if len(rows) == 0:
        return []
    cands = np.stack([rows, cols], axis=1).astype(float)
    kept = []
    for idx in np.argsort(-smoothed[rows, cols]):
        c = cands[idx]
        if all(np.linalg.norm(c - k) >= sep for k in kept):
            kept.append(c)
    support = [sum(int(m[int(round(c[0])), int(round(c[1]))]) for m in masks) for c in kept]
    resp = [smoothed[int(round(c[0])), int(round(c[1]))] for c in kept]
    rank = np.lexsort((-np.array(resp), -np.array(support)))
    return [support[i] for i in rank][:max_modes]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays", nargs="+", default=["275", "1725", "3613", "4520", "4664"])
    ap.add_argument("--label-root", default="/workspace/data/label/dst")
    ap.add_argument("--label-method", default="all_correct")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[2.0, 4.0, 6.0])
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.25, 0.35, 0.50])
    ap.add_argument("--seps", type=float, nargs="+", default=[8.0, 12.0, 16.0, 20.0])
    ap.add_argument("--default", type=float, nargs=3, default=[4.0, 0.35, 12.0],
                    metavar=("SIGMA", "THETA", "D"),
                    help="sigma values other than this one are run only at its theta and D")
    ap.add_argument("--max-modes", type=int, default=5)
    args = ap.parse_args()

    s0, t0, d0 = args.default
    configs = [(s, t, d) for s in args.sigmas for t in args.thetas for d in args.seps
               if s == s0 or (t == t0 and d == d0)]
    stats = {c: defaultdict(float) for c in configs}

    for rep in args.replays:
        coco = load_coco_gt(args.label_root, rep, args.label_method)
        h, w = image_size(coco)
        by_frame = gt_boxes_by_frame(coco)
        frames = sorted(by_frame)
        f0, span = frames[0], frames[-1] - frames[0] + 1
        cache = {}
        for f in frames:
            boxes = by_frame[f]
            key = boxes.tobytes()
            if key not in cache:
                masks = [box_mask(b, h, w) for b in boxes]
                cov = np.zeros((h, w))
                for m in masks:
                    cov += m
                cov /= max(1, len(masks))
                res = {}
                for s in args.sigmas:
                    sm = gaussian_filter(cov, sigma=s, mode="constant")
                    for c in configs:
                        if c[0] == s:
                            res[c] = ranked_support(masks, sm, c[1], c[2], args.max_modes) if sm.max() > 0 else []
                cache[key] = res
            prog = (f - f0) / max(1, span - 1)
            q = min(3, int(prog * 4))
            for c, sup in cache[key].items():
                n = len(sup)
                st = stats[c]
                st["frames"] += 1
                st["modes"] += n
                st["multi"] += n >= 2
                st["multi_s2"] += n >= 2 and sup[1] >= 2
                st["tie"] += n >= 2 and sup[0] == sup[1]
                st["tie22"] += n >= 2 and sup[0] == sup[1] and sup[1] >= 2
                st[f"q{q}_frames"] += 1
                st[f"q{q}_modes"] += n
        print(f"done {rep}: {len(frames)} frames, {len(cache)} unique viewport sets", file=sys.stderr, flush=True)

    print("sigma\ttheta\tD\tframes\tmeanM\tmulti\tmulti_s2\ttie\ttie22\tM_q1\tM_q2\tM_q3\tM_q4\tpeak_q")
    for c in configs:
        st = stats[c]
        n = st["frames"]
        qs = [st[f"q{i}_modes"] / st[f"q{i}_frames"] for i in range(4)]
        print("\t".join([f"{c[0]:g}", f"{c[1]:g}", f"{c[2]:g}", f"{int(n)}",
                         f"{st['modes'] / n:.3f}", f"{st['multi'] / n:.3f}", f"{st['multi_s2'] / n:.3f}",
                         f"{st['tie'] / n:.3f}", f"{st['tie22'] / n:.3f}"]
                        + [f"{x:.3f}" for x in qs] + [str(int(np.argmax(qs)) + 1)]))


if __name__ == "__main__":
    main()
