#!/usr/bin/env python3
"""
reeval_missing_frames.py
========================

Re-derive the Mask R-CNN / Mask R-CNN + KBRS numbers of the saliency-prior paper
(ToG-2026-0161) and measure how far the evaluator faults behind them moved them.

Those numbers were computed in December 2025 and January 2026, before two
evaluator fixes:

    8de59a3 / cf46716 (2026-08-20)  frames on which the model emitted no
                                    prediction were skipped; they are now scored
                                    0. Skipping them conditions every metric on
                                    the frames the model chose to answer.
    0782acd (2026-09-18)            the prediction centroid was read off the
                                    stored box, which inference clamps at the
                                    map edge; it is now measured at the fixed
                                    viewport size.

Each replay's ground truth and predictions are loaded once and scored three
ways, so the two fixes separate:

    a_legacy    --skip-missing-preds --legacy-centroid   should reproduce the
                                                          numbers on file
    b_centroid  --skip-missing-preds                      a -> b is 0782acd
    c_current   (neither)                                 b -> c is the
                                                          missing-frame fix;
                                                          c is the corrected
                                                          value

Every run also carries `old`, the replay-level CSV written at the time. If a_legacy
does not match it, something else has changed as well and the comparison says so
in its `legacy_check` column rather than being trusted.

Settings are those of the paper: kernel 20x12, grid 128x128, map 3456x3720,
labels all_correct, epoch 30, no score threshold. Metrics are averaged over frames
within a replay and then over replays, as the paper does.

Outputs, under --out (default /workspace/results/reeval/missing_frames_<date>):

    per_replay.csv   one row per run x replay x variant
    per_run.csv      replay means per run x variant, with the deltas
    summary.md       the two models aggregated over fold 1 and over all folds

Run it from the repository root:

    make reeval-missing-frames
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
import time
from typing import Dict, List

import pandas as pd

sys.path.insert(0, "/workspace/src")
from evaluate import compute_ic_for_replay, load_coco_gt, load_coco_preds  # noqa: E402

FOLD_REPLAYS: Dict[int, List[str]] = {
    1: ["275", "1725", "3613", "4520", "4664"],
    2: ["1559", "1628", "2351", "6219", "11251"],
    3: ["36", "212", "438", "522", "1660"],
}

# The runs behind the paper's main comparison. The KBRS fold-1 seed-123 cell was
# trained twice; the December run is the one the paper's period of reporting
# covers, and the February rerun is kept as a separate row, outside the means.
RUNS: List[str] = [
    "maskrcnn_win4_vanilla_fold1_s123_20251201_072032",
    "maskrcnn_win4_vanilla_fold1_s456_20251202_003952",
    "maskrcnn_win4_vanilla_fold1_s789_20251202_163217",
    "maskrcnn_win4_vanilla_fold2_s123_20251203_081925",
    "maskrcnn_win4_vanilla_fold2_s456_20251204_002252",
    "maskrcnn_win4_vanilla_fold2_s789_20251204_163313",
    "maskrcnn_win4_vanilla_fold3_s123_20251205_081306",
    "maskrcnn_win4_vanilla_fold3_s456_20251205_235535",
    "maskrcnn_win4_vanilla_fold3_s789_20251206_161212",
    "maskrcnn_win4_kbrs_fold1_s123_kbrs025_base_score_20251219_080334",
    "maskrcnn_win4_kbrs_fold1_s456_kbrs025_base_score_20260101_175816",
    "maskrcnn_win4_kbrs_fold1_s789_kbrs025_base_score_20251230_004200",
    "maskrcnn_win4_kbrs_fold2_s123_kbrs025_base_score_20251222_080550",
    "maskrcnn_win4_kbrs_fold2_s456_kbrs025_base_score_20251231_031613",
    "maskrcnn_win4_kbrs_fold2_s789_kbrs025_base_score_20251231_215319",
    "maskrcnn_win4_kbrs_fold3_s123_kbrs025_base_score_20251230_083127",
    "maskrcnn_win4_kbrs_fold3_s456_kbrs025_base_score_20251231_151012",
    "maskrcnn_win4_kbrs_fold3_s789_kbrs025_base_score_20260102_163643",
    "maskrcnn_win4_kbrs_fold1_s123_kbrs025_base_score_20260203_055642",
]
RERUNS = {"maskrcnn_win4_kbrs_fold1_s123_kbrs025_base_score_20260203_055642"}

VARIANTS = {
    "a_legacy": dict(skip_missing_preds=True, legacy_centroid=True),
    "b_centroid": dict(skip_missing_preds=True, legacy_centroid=False),
    "c_current": dict(skip_missing_preds=False, legacy_centroid=False),
}
METRICS = ["ic@000", "ic@030", "ic@050", "ic_ratio"]
METRIC_LABEL = {"ic@000": "@any", "ic@030": "@0.3", "ic@050": "@0.5", "ic_ratio": "IR"}


def parse_run(name: str) -> Dict[str, object]:
    m = re.search(r"maskrcnn_win4_(vanilla|kbrs)_fold(\d)_s(\d+)", name)
    if not m:
        raise ValueError(f"cannot parse run name: {name}")
    return {"model": m.group(1), "fold": int(m.group(2)), "seed": int(m.group(3))}


def load_old(results_root: str, run: str) -> pd.DataFrame:
    """The replay-level CSV written when the run was first evaluated."""
    path = os.path.join(results_root, f"{run}_e30.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "replay" not in df.columns:
        return pd.DataFrame()
    df["replay"] = pd.to_numeric(df["replay"], errors="coerce")
    df = df.dropna(subset=["replay"])
    df["replay"] = df["replay"].astype(int).astype(str)
    out = df[["replay"] + [m for m in METRICS if m in df.columns]].copy()
    if "num_images" in df.columns:
        out["evaluated_frames"] = df["num_images"].values
    out["variant"] = "old"
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-root", default="/workspace/predictions")
    p.add_argument("--label-root", default="/workspace/data/label/dst")
    p.add_argument("--results-root", default="/workspace/results")
    p.add_argument("--out", default=None)
    p.add_argument("--runs", nargs="*", default=None, help="Subset of run names (default: all)")
    args = p.parse_args()

    out_dir = args.out or os.path.join(
        args.results_root, "reeval", f"missing_frames_{dt.date.today():%Y%m%d}"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Writing to {out_dir}", flush=True)

    ic_args = argparse.Namespace(ic_kernel="20,12", ic_grid="128,128", ic_maxcoord="3456,3720", max_frames=0)
    runs = args.runs or RUNS
    rows: List[Dict[str, object]] = []
    gt_cache = {}

    for run in runs:
        meta = parse_run(run)
        print(f"\n[RUN] {run}", flush=True)
        old = load_old(args.results_root, run)
        for rec in old.to_dict("records"):
            rows.append({"run": run, **meta, **rec})

        for replay in FOLD_REPLAYS[meta["fold"]]:
            t0 = time.time()
            if replay not in gt_cache:
                gt_cache[replay] = load_coco_gt(args.label_root, replay, "all_correct")
            coco_gt = gt_cache[replay]
            preds = load_coco_preds(args.pred_root, run, 30, replay, "all_correct")
            if not preds:
                print(f"[!] no predictions for {run} replay {replay}; skipped", flush=True)
                continue
            for variant, flags in VARIANTS.items():
                r = compute_ic_for_replay(
                    replay_id=replay, mode="model", coco_gt=coco_gt, args=ic_args,
                    preds_all=preds, model_tag=f"{run}_{variant}", score_thresh=0.0, **flags,
                )
                rows.append({
                    "run": run, **meta, "replay": replay, "variant": variant,
                    **{m: r.get(m) for m in METRICS},
                    "total_frames": r.get("total_frames"),
                    "missing_preds": r.get("missing_preds"),
                    "evaluated_frames": r.get("evaluated_frames"),
                })
            del preds
            print(f"[RUN] {run} replay {replay} done in {time.time() - t0:.1f}s", flush=True)

        # Write as we go, so an interrupted run keeps what it has.
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "per_replay.csv"), index=False)

    per_replay = pd.DataFrame(rows)
    per_replay.to_csv(os.path.join(out_dir, "per_replay.csv"), index=False)

    # Replay means per run x variant (frames within a replay, then replays).
    agg = {m: "mean" for m in METRICS}
    agg.update({"missing_preds": "sum", "total_frames": "sum", "evaluated_frames": "sum"})
    agg = {k: v for k, v in agg.items() if k in per_replay.columns}
    per_run = (
        per_replay.groupby(["run", "model", "fold", "seed", "variant"], as_index=False)
        .agg(agg)
    )
    wide = per_run.pivot_table(index=["run", "model", "fold", "seed"], columns="variant", values=METRICS)
    wide.columns = [f"{m}__{v}" for m, v in wide.columns]
    wide = wide.reset_index()
    for m in METRICS:
        def col(v):
            return wide.get(f"{m}__{v}")
        if col("old") is not None and col("a_legacy") is not None:
            wide[f"{m}__legacy_check"] = (col("a_legacy") - col("old")).abs()
        if col("a_legacy") is not None and col("b_centroid") is not None:
            wide[f"{m}__d_centroid"] = col("b_centroid") - col("a_legacy")
        if col("b_centroid") is not None and col("c_current") is not None:
            wide[f"{m}__d_missing"] = col("c_current") - col("b_centroid")
        if col("old") is not None and col("c_current") is not None:
            wide[f"{m}__d_total"] = col("c_current") - col("old")
    frames = per_run[per_run["variant"] == "c_current"][["run", "missing_preds", "total_frames"]]
    wide = wide.merge(frames, on="run", how="left")
    wide["missing_rate"] = wide["missing_preds"] / wide["total_frames"]
    wide["rerun"] = wide["run"].isin(RERUNS)
    wide.to_csv(os.path.join(out_dir, "per_run.csv"), index=False)

    # Summary: the two models over fold 1 and over all folds, reruns excluded.
    main_runs = wide[~wide["rerun"]]
    lines = [
        "# Re-evaluation of the saliency-prior paper's Mask R-CNN / KBRS runs",
        "",
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}. Kernel 20x12, all_correct labels, epoch 30,",
        "no score threshold; frames averaged within a replay, then replays within a run,",
        "then runs. `old` is the CSV written at the time; `a_legacy` should match it.",
        "",
    ]
    for scope, sub in [("Fold 1, 3 seeds", main_runs[main_runs["fold"] == 1]), ("All folds, 9 runs", main_runs)]:
        lines += [f"## {scope}", "", "| Model | Metric | old | a_legacy | b_centroid | c_current | 0782acd | missing-frame fix | total |", "|---|---|---|---|---|---|---|---|---|"]
        for model in ["vanilla", "kbrs"]:
            s = sub[sub["model"] == model]
            for m in METRICS:
                def mean(suffix):
                    c = f"{m}__{suffix}"
                    return s[c].mean() if c in s.columns else float("nan")
                lines.append(
                    f"| {model} | {METRIC_LABEL[m]} | {mean('old'):.4f} | {mean('a_legacy'):.4f} | "
                    f"{mean('b_centroid'):.4f} | {mean('c_current'):.4f} | {mean('d_centroid'):+.4f} | "
                    f"{mean('d_missing'):+.4f} | {mean('d_total'):+.4f} |"
                )
        lines.append("")
        lines.append("KBRS minus vanilla, IR: " + ", ".join(
            f"{v} {sub[sub['model']=='kbrs'][f'ic_ratio__{v}'].mean() - sub[sub['model']=='vanilla'][f'ic_ratio__{v}'].mean():+.4f}"
            for v in ["old", "a_legacy", "b_centroid", "c_current"] if f"ic_ratio__{v}" in sub.columns
        ))
        lines.append("")
    lines += [
        "## Checks",
        "",
        f"- Largest |a_legacy - old| over runs and metrics: "
        f"{max(wide[c].max() for c in wide.columns if c.endswith('__legacy_check')):.6f}"
        " (should be ~0; if not, something besides the two fixes differs)",
        f"- Missing-prediction rate per run: min {wide['missing_rate'].min():.4%}, max {wide['missing_rate'].max():.4%}",
        "",
    ]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
