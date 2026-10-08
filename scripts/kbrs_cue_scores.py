#!/usr/bin/env python3
"""
kbrs_cue_scores.py
==================

The KBRS cue scores (density, centeredness, mixture) at the observers'
viewports and at each model's primary predicted viewport, raw and per-frame
normalized. Thesis tab:kbrs:scores (Section "Score Diagnostics at the
Predicted Viewport") and the matching table of the KBRS paper.

The scores are read from the KBRS cache that src/tools/kbrs/kbrs_cache.py
built from the INPUT frames (not from detector features): for each frame a
.npz with density / centeredness / mixture maps of a 12 x 20-tile window
slid over the 128 x 128 tile map ("valid" positions, so a 117 x 109 map whose
cell (y, x) is the window with top-left corner (x, y)). Mixture compares the
two players' entity channels of the input. The cache has a raw/ and a
norm01/ version; norm01 is raw min-max normalized per frame (Eq. kbrs:norm),
so only raw/ is read here and the normalization is applied in place, which
halves the reads.

Per frame:
  GT     the mean over the frame's observer viewports
  model  the highest-scoring prediction with score >= --score-threshold
         (frames without one are counted as unanswered and left out)
A viewport is looked up at its top-left corner, clipped into the map's valid
range, as src/tools/kbrs/kbrs_lookup.py does with --clip. Each source's value
is the mean over its frames, pooled over the replays.

Run on a worker (the cache is on the NAS, ~150k files per fold):

  make analysis SCRIPT=kbrs_cue_scores ARGS="--workers 16 \
      --model vanilla_s123=maskrcnn_win4_vanilla_fold1_s123_20251201_072032 \
      --model kbrs_s123=maskrcnn_win4_kbrs_fold1_s123_kbrs025_base_score_20251219_080334 \
      --out /workspace/results/kbrs_cue_scores/fold1.csv"
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis_common import (
    add_prediction_args,
    add_replay_args,
    load_regions,
    load_replay_gt,
    parse_model_spec,
)

CUES = ("density", "centeredness", "mixture")
NORM_EPS = 1e-6  # the epsilon of Eq. kbrs:norm


def minmax(a: np.ndarray) -> np.ndarray:
    """Per-frame min-max normalization of one cue map (Eq. kbrs:norm)."""
    return (a - a.min()) / (a.max() - a.min() + NORM_EPS)


def sample(maps: Dict[str, np.ndarray], corners: List[Tuple[float, float]]) -> Dict[str, float]:
    """Mean of each map over the given top-left corners (x, y), clipped into range."""
    h, w = maps["density"].shape
    xs = np.clip(np.array([int(x) for x, _ in corners]), 0, w - 1)
    ys = np.clip(np.array([int(y) for _, y in corners]), 0, h - 1)
    return {k: float(maps[k][ys, xs].mean()) for k in maps}


def score_frame(job) -> Optional[Tuple[int, Dict[str, Dict[str, float]]]]:
    """job = (npz path, frame, {source: [(x, y), ...]}) -> per source, raw and normalized cue values."""
    path, frame, corners_by_source = job
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=True) as z:
        raw = {k: z[k].astype(np.float64) for k in CUES}
    norm = {k: minmax(v) for k, v in raw.items()}
    out = {}
    for source, corners in corners_by_source.items():
        r = sample(raw, corners)
        n = sample(norm, corners)
        out[source] = {**{f"{k}_raw": r[k] for k in CUES}, **{f"{k}_norm": n[k] for k in CUES}}
    return frame, out


def frame_jobs(replay: str, gt, regions_by_model: Dict[str, dict], cache_root: str,
               score_threshold: float, max_frames: int, counts) -> List[tuple]:
    jobs = []
    frames = sorted(gt.viewports_by_frame)
    if max_frames > 0:
        frames = frames[:max_frames]
    for frame in frames:
        corners = {"GT": [(b[0], b[1]) for b in gt.viewports_by_frame[frame]]}
        for name, regions in regions_by_model.items():
            boxes, scores = regions.get(frame, (np.zeros((0, 4)), np.zeros(0)))
            if len(boxes) and scores[0] >= score_threshold:
                corners[name] = [(boxes[0][0], boxes[0][1])]
            else:
                counts[(replay, name)]["unanswered"] += 1
        jobs.append((os.path.join(cache_root, f"{replay}.rep", f"{frame}.npz"), frame, corners))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_replay_args(ap)
    add_prediction_args(ap)
    ap.add_argument("--model", action="append", default=[], metavar="NAME=MODEL[:EPOCH][@THRESHOLD]",
                    help="a prediction source; its primary region is scored")
    ap.add_argument("--cache-root", default="/workspace/data/input/dst/__kbrs_cache__/raw",
                    help="the raw KBRS cache (<cache-root>/<replay>.rep/<frame>.npz)")
    ap.add_argument("--score-threshold", type=float, default=0.5,
                    help="a model's top prediction counts only at or above this score")
    ap.add_argument("--max-frames", type=int, default=0, help="first N frames per replay (testing)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True, help="CSV of per-replay and pooled means")
    args = ap.parse_args()

    models = [parse_model_spec(s, args.epoch) for s in args.model]
    counts = defaultdict(lambda: defaultdict(int))
    sums = defaultdict(lambda: defaultdict(float))  # (replay, source) -> column -> sum

    for replay in args.replays:
        gt = load_replay_gt(args.label_root, replay, args.label_method)
        regions_by_model = {
            spec.name: load_regions(args.pred_root, spec.model, spec.epoch, replay,
                                    args.label_method, spec.threshold, None)
            for spec in models
        }
        jobs = frame_jobs(replay, gt, regions_by_model, args.cache_root,
                          args.score_threshold, args.max_frames, counts)
        with Pool(args.workers) as pool:
            for result in pool.imap_unordered(score_frame, jobs, chunksize=64):
                if result is None:
                    counts[(replay, "GT")]["missing_cache"] += 1
                    continue
                _, by_source = result
                for source, values in by_source.items():
                    counts[(replay, source)]["frames"] += 1
                    for col, v in values.items():
                        sums[(replay, source)][col] += v
        print(f"done {replay}: {len(jobs)} frames, missing cache {counts[(replay, 'GT')]['missing_cache']}",
              file=sys.stderr, flush=True)

    sources = ["GT"] + [spec.name for spec in models]
    cols = [f"{k}_raw" for k in CUES] + [f"{k}_norm" for k in CUES]
    rows = []
    for source in sources:
        for replay in list(args.replays) + ["pooled"]:
            keys = [(r, source) for r in args.replays] if replay == "pooled" else [(replay, source)]
            frames = sum(counts[k]["frames"] for k in keys)
            row = {"source": source, "replay": replay, "frames": frames,
                   "unanswered": sum(counts[k]["unanswered"] for k in keys),
                   "missing_cache": sum(counts[(k[0], "GT")]["missing_cache"] for k in keys)}
            for c in cols:
                row[c] = sum(sums[k][c] for k in keys) / frames if frames else float("nan")
            rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    pooled = df[df["replay"] == "pooled"].set_index("source")
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(pooled[["frames", "unanswered"] + cols].to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
