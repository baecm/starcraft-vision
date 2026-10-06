#!/usr/bin/env python3
"""
pool_size_convergence.py
========================

Thesis ch. 8, `sec:att:convergence`: does the mode structure of the aggregated
spectator distribution still change as spectators are added, or has it
levelled off by U = 5?

No larger pool exists, so the five spectators are subsampled down to
U in {2, 3, 4, 5}: every subset of each size (10 + 10 + 5 + 1), with modes
extracted from that subset's viewports by the same `extract_modes` and the
same parameters (sigma 4, theta 0.35, D 12) as `scripts/mode_disagreement.py`.
No model is involved; only the ground-truth viewports are read.

For each subset it records, pooled over frames (as the thesis tables are):
  agreement   - mean pairwise overlap between the subset's viewports, the
                intersection divided by the viewport area (all viewports share
                one size, so this is also intersection over either box)
  n_modes     - mean number of extracted modes, and the fraction with >= 2
  support_k   - mean support of the k-th ranked mode divided by U, so that
                subsets of different size are comparable
  held2       - fraction of frames whose second mode at least two of the
                subset hold (absolute count, as in the thesis tables)
  single2     - fraction of multi-modal frames whose second mode one holds
  tie, tie22  - fraction of frames whose top two modes have equal support,
                and of those, equal at two or more each

and per U it reports the mean over subsets and the standard error across
them. Observers are addressed by their position in each frame's annotation
list, the same convention as `analyse_method(drop_observer=...)`; a frame with
fewer viewports than a subset needs is skipped for that subset.

--sweep adds a U = 5 pass over a grid of D and theta, for the sensitivity of
the multi-modal fraction to the extraction settings (`sec:att:progression`).

Usage
-----
python scripts/pool_size_convergence.py \
    --replays 275 1725 3613 4520 4664 \
    --label-method all_correct \
    --outdir results/pool_size/fold1 \
    --workers 5 [--frame-stride 1] [--sweep]

Writes <outdir>/subsets.csv (one row per subset and replay, with the frame
count so any pooling can be redone), <outdir>/by_U.csv (pooled, mean and SE
over subsets), <outdir>/by_U_replay.csv (the same per replay, for the
replay-level check of `sec:att:replay`), and with --sweep <outdir>/sweep.csv.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from estimate import load_coco_gt  # noqa: E402
from metrics.modes import extract_modes, gt_boxes_by_frame, image_size  # noqa: E402

K_MAX = 3


def pairwise_agreement(obs: np.ndarray) -> float:
    """Mean over pairs of intersection / viewport area, boxes as [x, y, w, h]."""
    vals = []
    for a, b in itertools.combinations(obs, 2):
        iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
        ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
        vals.append(iw * ih / max(1e-9, a[2] * a[3]))
    return float(np.mean(vals)) if vals else np.nan


def _sums(n_obs: int) -> Dict[str, float]:
    d = {"frames": 0, "agreement": 0.0, "n_modes": 0.0, "multimodal": 0.0,
         "held2": 0.0, "single2": 0.0, "tie": 0.0, "tie22": 0.0}
    for k in range(1, K_MAX + 1):
        d[f"support_{k}"] = 0.0
        d[f"has_{k}"] = 0
    return d


def process_replay(job: Tuple) -> Tuple[List[dict], List[dict]]:
    replay, args = job
    coco = load_coco_gt(args.label_root, replay, args.label_method)
    height, width = image_size(coco)
    gt = gt_boxes_by_frame(coco)
    frames = sorted(gt)[:: args.frame_stride]
    n_total = args.num_observers

    subsets = [
        s for u in range(2, n_total + 1) for s in itertools.combinations(range(n_total), u)
    ]
    acc = {s: _sums(len(s)) for s in subsets}
    sweep_acc: Dict[Tuple[float, float], Dict[str, float]] = {}
    grid = [(d, t) for d in args.sweep_min_sep for t in args.sweep_rel_threshold] if args.sweep else []

    for frame in frames:
        obs_all = gt[frame]
        for s in subsets:
            if len(obs_all) <= max(s):
                continue
            obs = obs_all[list(s)]
            m = extract_modes(obs, height, width, sigma=args.sigma, min_sep=args.min_sep,
                              rel_threshold=args.rel_threshold, max_modes=args.max_modes)
            a = acc[s]
            u = len(s)
            a["frames"] += 1
            a["agreement"] += pairwise_agreement(obs)
            a["n_modes"] += len(m.support)
            a["multimodal"] += float(len(m.support) >= 2)
            if len(m.support) >= 2:
                s1, s2 = int(m.support[0]), int(m.support[1])
                a["held2"] += float(s2 >= 2)
                a["single2"] += float(s2 == 1)
                a["tie"] += float(s1 == s2)
                a["tie22"] += float(s1 == s2 and s2 >= 2)
            for k in range(1, K_MAX + 1):
                if len(m.support) >= k:
                    a[f"support_{k}"] += m.support[k - 1] / u
                    a[f"has_{k}"] += 1
        if grid and len(obs_all) >= n_total:
            obs = obs_all[:n_total]
            for d, t in grid:
                m = extract_modes(obs, height, width, sigma=args.sigma, min_sep=d,
                                  rel_threshold=t, max_modes=args.max_modes)
                b = sweep_acc.setdefault((d, t), {"frames": 0, "n_modes": 0.0, "multimodal": 0.0})
                b["frames"] += 1
                b["n_modes"] += len(m.support)
                b["multimodal"] += float(len(m.support) >= 2)

    rows = []
    for s, a in acc.items():
        rows.append({"replay": replay, "U": len(s), "subset": "-".join(map(str, s)), **a})
    sweep_rows = [{"replay": replay, "min_sep": d, "rel_threshold": t, **b}
                  for (d, t), b in sweep_acc.items()]
    print(f"[done] replay {replay}: {len(frames)} frames", flush=True)
    return rows, sweep_rows


def pool_subsets(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """Sum the per-replay accumulators of each subset, then turn them into means."""
    g = df.groupby(keys + ["U", "subset"], as_index=False).sum(numeric_only=True)
    out = g[keys + ["U", "subset", "frames"]].copy()
    out["agreement"] = g["agreement"] / g["frames"]
    out["n_modes"] = g["n_modes"] / g["frames"]
    out["multimodal"] = g["multimodal"] / g["frames"]
    for c in ("held2", "tie", "tie22"):
        out[c] = g[c] / g["frames"]
    # over the multi-modal frames, as the 61 % of the thesis
    out["single2"] = g["single2"] / g["multimodal"].replace(0, np.nan)
    for k in range(1, K_MAX + 1):
        # support of the k-th mode over the frames that have one
        out[f"support_{k}"] = g[f"support_{k}"] / g[f"has_{k}"].replace(0, np.nan)
    return out


def by_U(per_subset: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    metrics = ["agreement", "n_modes", "multimodal", "held2", "single2", "tie", "tie22"] + [f"support_{k}" for k in range(1, K_MAX + 1)]
    rows = []
    for key, g in per_subset.groupby(keys + ["U"]):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys + ["U"], key))
        row["n_subsets"] = len(g)
        row["frames"] = int(g["frames"].mean())
        for m in metrics:
            row[m] = g[m].mean()
            # spread over the subsets of one size; 0 at U = 5, where there is one
            row[f"{m}_se"] = g[m].std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays", type=str, nargs="+", required=True)
    ap.add_argument("--label-root", default="/workspace/data/label/dst")
    ap.add_argument("--label-method", default="all_correct")
    ap.add_argument("--outdir", default="results/pool_size")
    ap.add_argument("--num-observers", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--min-sep", type=float, default=12.0)
    ap.add_argument("--rel-threshold", type=float, default=0.35)
    ap.add_argument("--max-modes", type=int, default=5)
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="analyse every n-th frame; 1 keeps all, as the thesis tables do")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sweep-min-sep", type=float, nargs="+", default=[8.0, 12.0, 16.0, 20.0])
    ap.add_argument("--sweep-rel-threshold", type=float, nargs="+", default=[0.25, 0.35, 0.5])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    jobs = [(str(r), args) for r in args.replays]
    if args.workers > 1:
        with Pool(min(args.workers, len(jobs))) as pool:
            results = pool.map(process_replay, jobs)
    else:
        results = [process_replay(j) for j in jobs]

    raw = pd.DataFrame([r for rows, _ in results for r in rows])
    raw.to_csv(os.path.join(args.outdir, "subsets.csv"), index=False)

    pooled = by_U(pool_subsets(raw, []), [])
    pooled.to_csv(os.path.join(args.outdir, "by_U.csv"), index=False)
    by_U(pool_subsets(raw, ["replay"]), ["replay"]).to_csv(
        os.path.join(args.outdir, "by_U_replay.csv"), index=False)

    cols = ["U", "n_subsets", "frames", "agreement", "agreement_se", "n_modes", "n_modes_se",
            "multimodal", "multimodal_se", "held2", "held2_se", "single2", "tie", "tie22"] + [f"support_{k}" for k in range(1, K_MAX + 1)]
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(pooled[cols].to_string(index=False))

    if args.sweep:
        sw = pd.DataFrame([r for _, rows in results for r in rows])
        g = sw.groupby(["min_sep", "rel_threshold"], as_index=False).sum(numeric_only=True)
        g["n_modes"] = g["n_modes"] / g["frames"]
        g["multimodal"] = g["multimodal"] / g["frames"]
        g = g.drop(columns=[c for c in g.columns if c == "replay"])
        g.to_csv(os.path.join(args.outdir, "sweep.csv"), index=False)
        with pd.option_context("display.width", 200, "display.precision", 4):
            print(g.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
