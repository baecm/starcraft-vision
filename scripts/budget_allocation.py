"""How each method spends its region budget across scene complexity.

Reads the per-frame CSVs that mode_disagreement.py writes and answers a
question the aggregate metrics hide: two methods can emit the same mean number
of regions and still allocate them oppositely across the test set.

The motivation is the count responsiveness beta_n. A proposal detector's
beta_n is negative at every confidence threshold, meaning it emits *fewer*
regions on frames whose target has more modes. beta_n on its own is a
statistic; this script measures what it costs. A frame whose target has n
ranked modes and for which a method emits fewer than n regions has observers
that cannot be served no matter where those regions are placed - the failure is
capacity, not localisation. Conditioning everything on n separates the two and
usually produces a crossover: the method with the larger budget on easy frames
wins there and loses on the hard ones.

Two cautions, both of which the output marks rather than hides:

- A method whose decoder caps at K regions cannot exceed K, so its
  under-emission rate at n > K is 1 by construction and says nothing about the
  method. Rows where the cap binds are flagged and excluded from the summary.
- OC@delta is only comparable across methods when each is scored on the number
  of regions it actually emitted, so the column used is OC at
  K = min(n_pred, 5) per frame rather than a fixed K.

Usage:
  python3 scripts/budget_allocation.py --dir results/mode_disagreement/th_sweep \
      [--methods director prior_th07] [--k-cap director=3] [--out summary.csv]
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional

MAX_N = 5


def _read(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _oc_at(row: Dict[str, str], k: int) -> Optional[float]:
    col = f"OC{k}@0.5"
    if col not in row or row[col] in (None, ""):
        return None
    return float(row[col])


def analyse(rows: List[Dict[str, str]], k_cap: Optional[int]) -> Dict:
    frames = defaultdict(int)
    n_pred_sum = defaultdict(float)
    under = defaultdict(int)
    deficit = defaultdict(float)
    oc_sum = defaultdict(float)
    oc_n = defaultdict(int)
    missing_oc = 0

    for row in rows:
        n = int(float(row["n_modes"]))
        p = int(float(row["n_pred"]))
        if n < 1 or n > MAX_N:
            continue
        frames[n] += 1
        n_pred_sum[n] += p
        if p < n:
            under[n] += 1
            deficit[n] += n - p
        oc = _oc_at(row, max(1, min(p, MAX_N)))
        if oc is None:
            missing_oc += 1
        else:
            oc_sum[n] += oc
            oc_n[n] += 1

    per_n = {}
    for n in range(1, MAX_N + 1):
        if not frames[n]:
            continue
        per_n[n] = {
            "frames": frames[n],
            "n_pred": n_pred_sum[n] / frames[n],
            "under_rate": under[n] / frames[n],
            "deficit": deficit[n] / frames[n],
            "oc": (oc_sum[n] / oc_n[n]) if oc_n[n] else float("nan"),
            # the decoder cannot emit more than its cap, so at n > cap the
            # under-emission rate is 1 whatever the model does
            "capped": k_cap is not None and n > k_cap,
        }

    # Responsiveness, the weighted slope of E[n_pred | n] on n, so the table and
    # the headline statistic are computed from the same rows.
    pts = [(n, v["n_pred"], v["frames"]) for n, v in per_n.items()]
    w = sum(f for _, _, f in pts)
    mx = sum(n * f for n, _, f in pts) / w
    my = sum(p * f for _, p, f in pts) / w
    num = sum(f * (n - mx) * (p - my) for n, p, f in pts)
    den = sum(f * (n - mx) ** 2 for n, _, f in pts)

    return {
        "per_n": per_n,
        "beta_n": num / den if den else float("nan"),
        "n_pred": sum(v["n_pred"] * v["frames"] for v in per_n.values()) / w,
        "frames": w,
        "missing_oc": missing_oc,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="a mode_disagreement output directory")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="method labels to include; default is all found")
    ap.add_argument("--k-cap", nargs="*", default=[],
                    help="label=K for methods whose decoder caps at K regions, "
                         "e.g. director=3")
    ap.add_argument("--out", default=None, help="write the table to this CSV")
    args = ap.parse_args()

    caps: Dict[str, int] = {}
    for item in args.k_cap:
        label, _, k = item.partition("=")
        caps[label] = int(k)

    found = {}
    for path in sorted(glob.glob(os.path.join(args.dir, "frames_*.csv"))):
        label = os.path.basename(path)[len("frames_"):-len(".csv")]
        if args.methods and label not in args.methods:
            continue
        found[label] = path
    if not found:
        print(f"no frames_*.csv in {args.dir}")
        return 1

    results = {}
    for label, path in found.items():
        rows = _read(path)
        results[label] = analyse(rows, caps.get(label))

    out_rows = []
    for label, res in results.items():
        cap = caps.get(label)
        cap_note = f", decoder cap K={cap}" if cap else ""
        print(f"\n{label}  (frames={res['frames']:,}  "
              f"mean |P|={res['n_pred']:.2f}  beta_n={res['beta_n']:+.3f}"
              f"{cap_note})")
        if res["missing_oc"]:
            print(f"  note: {res['missing_oc']:,} rows had no OC column at the "
                  f"emitted K; they are excluded from the OC column")
        print(f"  {'n':>2} {'frames':>8} {'|P|':>6} {'under':>7} "
              f"{'deficit':>8} {'OC@0.5':>8}")
        for n, v in sorted(res["per_n"].items()):
            flag = "  (capped)" if v["capped"] else ""
            print(f"  {n:>2} {v['frames']:>8,} {v['n_pred']:>6.2f} "
                  f"{100 * v['under_rate']:>6.1f}% {v['deficit']:>8.2f} "
                  f"{v['oc']:>8.3f}{flag}")
            out_rows.append({
                "method": label, "n_modes": n, "frames": v["frames"],
                "n_pred": round(v["n_pred"], 4),
                "under_rate": round(v["under_rate"], 4),
                "deficit": round(v["deficit"], 4),
                "oc": round(v["oc"], 4), "capped": int(v["capped"]),
                "beta_n": round(res["beta_n"], 4),
            })

    if len(results) == 2:
        (a, ra), (b, rb) = results.items()
        print(f"\n{a} vs {b}, on the n where neither decoder cap binds:")
        for n in sorted(set(ra["per_n"]) & set(rb["per_n"])):
            va, vb = ra["per_n"][n], rb["per_n"][n]
            if va["capped"] or vb["capped"]:
                continue
            ratio = (vb["under_rate"] / va["under_rate"]
                     if va["under_rate"] > 0 else math.inf)
            print(f"  n={n}: under-emission {100 * va['under_rate']:.1f}% vs "
                  f"{100 * vb['under_rate']:.1f}%  ({ratio:.1f}x)   "
                  f"OC {va['oc']:.3f} vs {vb['oc']:.3f} "
                  f"({va['oc'] - vb['oc']:+.3f})")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            wr.writeheader()
            wr.writerows(out_rows)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
