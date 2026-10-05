#!/usr/bin/env python3
"""
split_by_second_mode.py
=======================

Region count, count responsiveness and per-spectator coverage, split by the
support of the second attention mode. Reads the per-frame records that
`mode_disagreement.py` writes (`<outdir>/frames_<method>.csv`).

Frames fall into three groups:
  uni  one mode
  m1   two or more modes, the second held by one spectator
  m2   two or more modes, the second held by at least two spectators

beta_n is the slope of the emitted region count on the number of modes
(equal to the weighted fit of Eq. countslope), computed on all frames, on
uni + m2 and on uni + m1. A declined frame counts as zero regions and scores
0 on IR and OC3, as everywhere else in the evaluation.

Usage
-----
python scripts/split_by_second_mode.py \
    results/mode_disagreement/v6_sweep_s123/frames_full.csv \
    results/mode_disagreement/v6_sweep_s123/frames_prior_th09.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", nargs="+", help="frames_<method>.csv files")
    args = ap.parse_args()

    for path in args.frames:
        df = pd.read_csv(path)
        answered = df["answered"].astype(bool)
        n_pred = np.where(answered, df["n_pred"], 0).astype(float)
        # Coverage-penalized: an unanswered frame scores 0 (its IR / OC cells are empty).
        for col in ("IR", "OC3@0.5"):
            df[col] = df[col].where(answered, 0.0).fillna(0.0)
        n_modes = df["n_modes"].to_numpy(dtype=float)
        group = np.where(n_modes <= 1, "uni", np.where(df["support_top2"] >= 2, "m2", "m1"))

        print(path)
        print(f"  beta_all={slope(n_modes, n_pred):+.3f}"
              f"  beta_uni+m2={slope(n_modes[group != 'm1'], n_pred[group != 'm1']):+.3f}"
              f"  beta_uni+m1={slope(n_modes[group != 'm2'], n_pred[group != 'm2']):+.3f}")
        for g in ("uni", "m1", "m2"):
            sel = group == g
            print(f"  {g}: frames={sel.sum()} share={sel.mean():.3f} n_pred={n_pred[sel].mean():.2f}"
                  f" IR={df['IR'][sel].mean():.3f} OC3={df['OC3@0.5'][sel].mean():.3f}")


if __name__ == "__main__":
    main()
