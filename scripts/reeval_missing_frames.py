#!/usr/bin/env python3
"""
reeval_missing_frames.py
========================

Re-derive single-region numbers reported from earlier versions of the evaluator,
and measure how far each evaluator fix moved them.

Four fixes are separated:

    8de59a3 / cf46716 (2026-08-20)  frames on which the model emitted no
                                    prediction were skipped. Skipping them
                                    conditions every metric on the frames the
                                    model chose to answer. The fix gave them a
                                    -9999 sentinel, meant to score 0.
    0782acd (2026-09-18)            the prediction centroid was read off the
                                    stored box, which inference clamps at the
                                    map edge; it is now measured at the fixed
                                    viewport size.
    missing-as-corner (2026-10-01)  the -9999 sentinel went through the
                                    evaluator's clip and was scored as a
                                    viewport at the map's top-left corner, not
                                    as 0. Such frames now score exactly 0.
    anchor (2026-10-01)             the kernel evaluator places a window at an
                                    agent's position as its top-left corner,
                                    but was passed each annotation's centroid.
                                    Every window moved by half a viewport, and
                                    past the far edge the clip stacked windows
                                    onto one position, inflating overlap. Both
                                    prediction and observers now pass corners.

Each replay's ground truth and predictions are loaded once and scored five ways:

    a_legacy    skip missing, legacy centroid,       should reproduce the CSV
                centroid anchor                      written at the time
    b_centroid  skip missing, centroid anchor        a -> b is 0782acd
    c_corner    missing as the corner, centroid      b -> c is the August fix as
                anchor                               it was implemented
    d_zero      missing scored 0, centroid anchor    c -> d is the corner fix
    e_current   missing scored 0, corner anchor      d -> e is the anchor fix;
                                                     e is the corrected value

e_current should agree with scripts/mode_disagreement.py on the same run, which
measures the same ratio from the stored boxes directly; the `ch9` set exists
partly to check that.

Every run with a CSV on file also carries `old`. If a_legacy does not match it,
something else has changed too, and `legacy_check` says so.

Two sets of runs:

    --set kbrs   the saliency-prior paper's (ToG-2026-0161) main comparison:
                 9 Mask R-CNN and 9 Mask R-CNN + KBRS runs, plus the February
                 KBRS rerun of fold 1 seed 123, kept outside the means
    --set ch9    the Director-CenterNet comparison (thesis ch. 9): the full
                 model and Mask R-CNN at tau = 0.5 and 0.9, fold 1, 3 seeds.
                 No old CSVs exist for these; only c -> d is of interest

A run is named `<run>` or `<run>@<tau>`; `@<tau>` reads predictions from
`model_030_th<tau>`. Settings are fixed: kernel 20x12, grid 128x128, map
3456x3720, labels all_correct, epoch 30, no further score threshold. Metrics are
averaged over frames within a replay and then over replays.

Outputs, under --out (default /workspace/results/reeval/<set>_<date>):

    per_replay.csv   one row per run x replay x variant
    per_run.csv      replay means per run x variant, with the deltas
    summary.md       aggregated per model, with checks

    make reeval-missing-frames ARGS="--set kbrs"
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, "/workspace/src")
from evaluate import compute_ic_for_replay, load_coco_gt, load_coco_preds  # noqa: E402

FOLD_REPLAYS: Dict[int, List[str]] = {
    1: ["275", "1725", "3613", "4520", "4664"],
    2: ["1559", "1628", "2351", "6219", "11251"],
    3: ["36", "212", "438", "522", "1660"],
}

RUN_SETS: Dict[str, List[str]] = {
    "kbrs": [
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
    ],
    "ch9": [
        *[f"dc_full_b16_f1_s{s}_v6" for s in (123, 456, 789)],
        *[f"maskrcnn_win4_vanilla_f1_s{s}_v6@0.5" for s in (123, 456, 789)],
        *[f"maskrcnn_win4_vanilla_f1_s{s}_v6@0.9" for s in (123, 456, 789)],
    ],
}
# Trained twice; the December run is the one the paper's reporting period covers.
RERUNS = {"maskrcnn_win4_kbrs_fold1_s123_kbrs025_base_score_20260203_055642"}

VARIANTS = {
    "a_legacy": dict(skip_missing_preds=True, legacy_centroid=True, missing_as_corner=False, legacy_anchor=True),
    "b_centroid": dict(skip_missing_preds=True, legacy_centroid=False, missing_as_corner=False, legacy_anchor=True),
    "c_corner": dict(skip_missing_preds=False, legacy_centroid=False, missing_as_corner=True, legacy_anchor=True),
    "d_zero": dict(skip_missing_preds=False, legacy_centroid=False, missing_as_corner=False, legacy_anchor=True),
    "e_current": dict(skip_missing_preds=False, legacy_centroid=False, missing_as_corner=False, legacy_anchor=False),
}
DELTAS = {  # name: (to, from)
    "d_centroid": ("b_centroid", "a_legacy"),
    "d_missing": ("c_corner", "b_centroid"),
    "d_corner": ("d_zero", "c_corner"),
    "d_anchor": ("e_current", "d_zero"),
    "d_total": ("e_current", "old"),
}
METRICS = ["ic@000", "ic@030", "ic@050", "ic_ratio"]
METRIC_LABEL = {"ic@000": "@any", "ic@030": "@0.3", "ic@050": "@0.5", "ic_ratio": "IR"}


def split_spec(spec: str) -> Tuple[str, Optional[str]]:
    run, _, tau = spec.partition("@")
    return run, (tau or None)


def parse_run(spec: str) -> Dict[str, object]:
    run, tau = split_spec(spec)
    fold = re.search(r"_f(?:old)?(\d)_", run)
    seed = re.search(r"_s(\d{3})(?:_|$)", run)
    if not fold or not seed:
        raise ValueError(f"cannot find fold/seed in run name: {run}")
    if run.startswith("dc_"):
        model = "director"
    elif "_kbrs_" in run:
        model = "kbrs"
    else:
        model = "maskrcnn"
    if tau:
        model = f"{model}@{tau}"
    return {"model": model, "fold": int(fold.group(1)), "seed": int(seed.group(1)), "tau": tau}


def load_old(results_root: str, run: str) -> pd.DataFrame:
    """The replay-level CSV written when the run was first evaluated, if any."""
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
    p.add_argument("--set", choices=sorted(RUN_SETS), default="kbrs")
    p.add_argument("--runs", nargs="*", default=None, help="Run specs <run>[@tau] (default: the whole --set)")
    p.add_argument("--pred-root", default="/workspace/predictions")
    p.add_argument("--label-root", default="/workspace/data/label/dst")
    p.add_argument("--results-root", default="/workspace/results")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_dir = args.out or os.path.join(args.results_root, "reeval", f"{args.set}_{dt.date.today():%Y%m%d}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Writing to {out_dir}", flush=True)

    ic_args = argparse.Namespace(ic_kernel="20,12", ic_grid="128,128", ic_maxcoord="3456,3720", max_frames=0)
    specs = args.runs or RUN_SETS[args.set]
    rows: List[Dict[str, object]] = []
    gt_cache = {}

    for spec in specs:
        run, tau = split_spec(spec)
        meta = parse_run(spec)
        print(f"\n[RUN] {spec}", flush=True)
        if tau is None:
            for rec in load_old(args.results_root, run).to_dict("records"):
                rows.append({"run": spec, **meta, **rec})

        for replay in FOLD_REPLAYS[meta["fold"]]:
            t0 = time.time()
            if replay not in gt_cache:
                gt_cache[replay] = load_coco_gt(args.label_root, replay, "all_correct")
            coco_gt = gt_cache[replay]
            preds = load_coco_preds(args.pred_root, run, 30, replay, "all_correct", score_threshold=tau)
            if not preds:
                print(f"[!] no predictions for {spec} replay {replay}; skipped", flush=True)
                continue
            for variant, flags in VARIANTS.items():
                r = compute_ic_for_replay(
                    replay_id=replay, mode="model", coco_gt=coco_gt, args=ic_args,
                    preds_all=preds, model_tag=f"{spec}_{variant}", score_thresh=0.0, **flags,
                )
                rows.append({
                    "run": spec, **meta, "replay": replay, "variant": variant,
                    **{m: r.get(m) for m in METRICS},
                    "total_frames": r.get("total_frames"),
                    "missing_preds": r.get("missing_preds"),
                    "evaluated_frames": r.get("evaluated_frames"),
                })
            del preds
            print(f"[RUN] {spec} replay {replay} done in {time.time() - t0:.1f}s", flush=True)

        # Write as we go, so an interrupted run keeps what it has.
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "per_replay.csv"), index=False)

    per_replay = pd.DataFrame(rows)
    per_replay.to_csv(os.path.join(out_dir, "per_replay.csv"), index=False)

    # Replay means per run x variant: frames within a replay, then replays.
    keys = ["run", "model", "fold", "seed", "variant"]
    per_run = per_replay.groupby(keys, as_index=False, dropna=False)[METRICS].mean()
    frames = (
        per_replay[per_replay["variant"] == "e_current"]
        .groupby("run", as_index=False)[["missing_preds", "total_frames"]].sum()
    )
    wide = per_run.pivot_table(index=["run", "model", "fold", "seed"], columns="variant", values=METRICS)
    wide.columns = [f"{m}__{v}" for m, v in wide.columns]
    wide = wide.reset_index()
    for m in METRICS:
        old_c, leg_c = f"{m}__old", f"{m}__a_legacy"
        if old_c in wide.columns and leg_c in wide.columns:
            wide[f"{m}__legacy_check"] = (wide[leg_c] - wide[old_c]).abs()
        for name, (to, frm) in DELTAS.items():
            if f"{m}__{to}" in wide.columns and f"{m}__{frm}" in wide.columns:
                wide[f"{m}__{name}"] = wide[f"{m}__{to}"] - wide[f"{m}__{frm}"]
    wide = wide.merge(frames, on="run", how="left")
    wide["missing_rate"] = wide["missing_preds"] / wide["total_frames"]
    wide["rerun"] = wide["run"].isin(RERUNS)
    wide.to_csv(os.path.join(out_dir, "per_run.csv"), index=False)

    # Summary per model, reruns excluded.
    main_runs = wide[~wide["rerun"]]
    cols = ["old", "a_legacy", "b_centroid", "c_corner", "d_zero", "e_current", "d_centroid", "d_missing", "d_corner", "d_anchor", "d_total"]
    lines = [
        f"# Re-evaluation: set `{args.set}`",
        "",
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}. Kernel 20x12, all_correct labels, epoch 30.",
        "Frames averaged within a replay, then replays within a run, then runs.",
        "`old` is the CSV written at the time; `a_legacy` should match it. `e_current` is the corrected value.",
        "Deltas: d_centroid = b - a (0782acd), d_missing = c - b (August fix as implemented),",
        "d_corner = d - c (corner fix), d_anchor = e - d (anchor fix), d_total = e - old.",
        "",
    ]
    scopes = [("Fold 1", main_runs[main_runs["fold"] == 1])]
    if main_runs["fold"].nunique() > 1:
        scopes.append(("All folds", main_runs))
    for scope, sub in scopes:
        lines += [
            f"## {scope}", "",
            "| Model | runs | Metric | " + " | ".join(cols) + " |",
            "|---|---|---|" + "---|" * len(cols),
        ]
        for model in sorted(sub["model"].unique()):
            s = sub[sub["model"] == model]
            for m in METRICS:
                cells = []
                for c in cols:
                    k = f"{m}__{c}"
                    v = s[k].mean() if k in s.columns else float("nan")
                    cells.append("—" if pd.isna(v) else (f"{v:+.4f}" if c in DELTAS else f"{v:.4f}"))
                lines.append(f"| {model} | {len(s)} | {METRIC_LABEL[m]} | " + " | ".join(cells) + " |")
        if {"kbrs", "maskrcnn"} <= set(sub["model"]):
            k, v = sub[sub["model"] == "kbrs"], sub[sub["model"] == "maskrcnn"]
            diffs = [
                f"{c} {k[f'ic_ratio__{c}'].mean() - v[f'ic_ratio__{c}'].mean():+.4f}"
                for c in ["old", "a_legacy", "b_centroid", "c_corner", "d_zero", "e_current"] if f"ic_ratio__{c}" in sub.columns
            ]
            lines += ["", "KBRS minus Mask R-CNN, IR: " + ", ".join(diffs)]
        lines.append("")

    checks = [c for c in wide.columns if c.endswith("__legacy_check")]
    lines += ["## Checks", ""]
    if checks and wide[checks].notna().any().any():
        lines.append(
            f"- Largest |a_legacy - old| over runs and metrics: {wide[checks].max().max():.6f} "
            "(should be ~0; if not, something besides these fixes differs)"
        )
    lines.append("- Missing-prediction rate per run:")
    for r in wide.sort_values("run").itertuples():
        if pd.isna(r.total_frames):
            lines.append(f"  - {r.run}: not evaluated")
            continue
        lines.append(f"  - {r.run}: {r.missing_rate:.4%} ({int(r.missing_preds)} of {int(r.total_frames)})")
    lines.append("")

    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
