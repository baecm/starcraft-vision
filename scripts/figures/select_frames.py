"""
figures/select_frames.py
========================

Frame selection for the qualitative figures. Nothing here draws: each
subcommand prints candidates, with the numbers a caption needs to say how the
example was chosen.

  select              candidate frames for compare/heatmap and windows for
                      trajectory, from the frames_<name>.csv that
                      mode_disagreement.py wrote for both methods
  select-supervision  frames that show the supervision construction legibly,
                      from the ground truth alone
  select-single       candidates for single_failure, from one method's CSV

  make figure-fg FIG=select_frames ARGS="select \
      --baseline-csv /workspace/results/mode_disagreement/fold1/frames_maskrcnn.csv \
      --director-csv /workspace/results/mode_disagreement/fold1/frames_director.csv"

Same as the `select`, `select-supervision` and `select-single` subcommands of
qualitative_figures.py.
"""

from __future__ import annotations

import argparse
import os
import sys

_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import numpy as np
import pandas as pd

from figures.common import common_parser, frame_modes, load_gt

# --------------------------------------------------------------------------
# select
# --------------------------------------------------------------------------

SELECT_HELP = "list candidate frames and windows"


def add_select_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--baseline-csv", required=True, help="frames_<baseline>.csv from mode_disagreement.py")
    p.add_argument("--director-csv", required=True, help="frames_<director>.csv from mode_disagreement.py")
    p.add_argument("--n-modes", type=int, default=3)
    p.add_argument("--quantile", type=float, default=0.5,
                   help="quantile of the per-frame OC difference to draw from")
    p.add_argument("--min-run", type=int, default=6, help="shortest tie run listed for (3)")
    p.add_argument("--top", type=int, default=15)


def cmd_select(args) -> None:
    oc = f"OC{args.k}@{args.delta}"
    keep = ["replay", "frame", "n_modes", "margin", "answered", "n_pred", oc, "top2_flip"]
    base = pd.read_csv(args.baseline_csv, usecols=lambda c: c in keep)
    dire = pd.read_csv(args.director_csv, usecols=lambda c: c in keep)
    df = base.merge(dire, on=["replay", "frame"], suffixes=("_base", "_dir"))
    df = df.rename(columns={"n_modes_base": "n_modes", "margin_base": "margin"})
    print(f"[select] {len(df)} frames in both CSVs", flush=True)

    # ---- (1)/(2): one frame with the requested number of modes, both answered
    pool = df[(df["n_modes"] == args.n_modes) & (df["answered_base"] == 1)
              & (df["answered_dir"] == 1) & (df["n_pred_dir"] >= 2)].copy()
    if pool.empty:
        print(f"[select] no frame with n_modes={args.n_modes} answered by both")
    else:
        pool["diff"] = pool[f"{oc}_dir"] - pool[f"{oc}_base"]
        target = pool["diff"].quantile(args.quantile)
        pool["dist"] = (pool["diff"] - target).abs()
        print()
        print(f"(1)/(2) pool: {len(pool)} frames with n_modes={args.n_modes}, both answered, "
              f"Director emitting >= 2 regions")
        print(f"  {oc} Director - baseline: mean {pool['diff'].mean():+.3f}, "
              f"median {pool['diff'].median():+.3f}; Director ahead on "
              f"{(pool['diff'] > 0).mean():.1%}, tied {(pool['diff'] == 0).mean():.1%}, "
              f"behind {(pool['diff'] < 0).mean():.1%}")
        print(f"  frames at the {args.quantile:.0%} quantile of the difference ({target:+.3f}):")
        cols = ["replay", "frame", "margin", f"{oc}_base", f"{oc}_dir", "n_pred_base", "n_pred_dir"]
        print(pool.sort_values(["dist", "replay", "frame"]).head(args.top)[cols].to_string(index=False))
        print("  Caption: say the frame was drawn at this quantile and give the three shares above.")

    # ---- (3): runs of consecutive tie frames
    runs = []
    for replay, g in df.sort_values("frame").groupby("replay"):
        step = g["frame"].diff()
        modal = step.mode().iloc[0] if len(step.mode()) else np.nan
        tie = (g["margin"] == 0).to_numpy()
        adjacent = (step == modal).to_numpy()
        frames = g["frame"].to_numpy()
        fb = g["top2_flip_base"].to_numpy()
        fd = g["top2_flip_dir"].to_numpy()
        start = None
        for i in range(len(g) + 1):
            inside = i < len(g) and tie[i] and (start is None or adjacent[i])
            if inside and start is None:
                start = i
            elif not inside and start is not None:
                n = i - start
                if n >= args.min_run:
                    runs.append({
                        "replay": replay, "start": int(frames[start]), "end": int(frames[i - 1]),
                        "frames": n,
                        "flips_base": int(np.nansum(fb[start:i])),
                        "flips_dir": int(np.nansum(fd[start:i])),
                    })
                start = i if (i < len(g) and tie[i]) else None
    print()
    if not runs:
        print(f"(3) no run of >= {args.min_run} consecutive margin-0 frames")
        return
    rdf = pd.DataFrame(runs)
    all_tie = df[df["margin"] == 0]
    print(f"(3) {len(rdf)} runs of >= {args.min_run} consecutive margin-0 frames. "
          f"Fold-wide top-2 flip rate on margin-0 frames: baseline "
          f"{all_tie['top2_flip_base'].mean():.4f}, Director {all_tie['top2_flip_dir'].mean():.4f}")
    rdf["rate_base"] = rdf["flips_base"] / rdf["frames"]
    print("  longest runs (pad the window with --pad frames either side when plotting):")
    print(rdf.sort_values("frames", ascending=False).head(args.top).to_string(index=False))
    print("  Caption: give this window's flip counts next to the fold-wide rates, so the "
          "example is read against the average and not as one.")


# --------------------------------------------------------------------------
# select-supervision
# --------------------------------------------------------------------------

SELECT_SUPERVISION_HELP = "rank frames by how legibly they show the supervision"


def add_select_supervision_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--replays", nargs="+", required=True)
    p.add_argument("--min-modes", type=int, default=3)
    p.add_argument("--min-margin", type=int, default=1,
                   help="support of the Top-1 mode minus the Top-2's; 1 excludes ties")
    # Not --min-sep: that is the extraction parameter D, the floor the mode
    # finder enforces. This is the separation the frame actually achieved.
    p.add_argument("--min-gap", type=float, default=30.0,
                   help="smallest pairwise distance between the extracted modes, in tiles")
    p.add_argument("--step", type=int, default=20, help="frame sampling stride")
    p.add_argument("--per-replay", type=int, default=3)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--out", default=None, help="write the full ranked list here")


def cmd_select_supervision(args) -> None:
    """Rank frames by how legibly they show the supervision construction.

    What makes a frame readable in the `supervision` figure is not how well any
    model does on it - no model appears - but the geometry of the ground truth:
    enough modes to show a ranking, a Top-1 that is not tied with the next one,
    and modes far enough apart that the panels do not look like one blob. Those
    come from the observer viewports alone, so this reads no predictions and
    needs no GPU.

    Frames are sampled at `--step`, because consecutive frames are near
    duplicates and scoring all of them buys nothing.
    """
    rows = []
    for replay in args.replays:
        replay = str(replay)
        gt, height, width, size_hw = load_gt(args, replay)
        frames = sorted(gt)[::args.step]
        for f in frames:
            obs = gt[f]
            m = frame_modes(obs, height, width, args)
            n = len(m.centers)
            if n < args.min_modes:
                continue
            sup = [int(s) for s in m.support]
            margin = sup[0] - sup[1] if n >= 2 else sup[0]
            if margin < args.min_margin:
                continue
            sep = min(float(np.linalg.norm(m.centers[i] - m.centers[j]))
                      for i in range(n) for j in range(i + 1, n))
            if sep < args.min_gap:
                continue
            rows.append({"replay": replay, "frame": int(f), "n_modes": n,
                         "support": "/".join(str(s) for s in sup),
                         "margin": margin, "min_sep": round(sep)})
        print(f"  {replay}: {len(frames)} frames sampled", flush=True)

    if not rows:
        print("\nno frame met the criteria; loosen --min-gap or --min-margin")
        return
    df = pd.DataFrame(rows).sort_values(["min_sep", "margin"], ascending=False)

    # One per replay first, so the shortlist is not five frames of one game.
    best = df.groupby("replay", as_index=False).head(args.per_replay)
    best = best.sort_values(["min_sep", "margin"], ascending=False).head(args.top)
    pd.set_option("display.width", 200)
    print(f"\n{len(df)} frames met the criteria; best {len(best)}, "
          f"at most {args.per_replay} per replay:\n")
    print(best.to_string(index=False))
    print("\nRender one with:  supervision --replay R --frame F")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"wrote the full list to {args.out}")


# --------------------------------------------------------------------------
# select-single
# --------------------------------------------------------------------------

SELECT_SINGLE_HELP = "candidates for single-failure, from one frames CSV"


def add_select_single_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--csv", required=True, help="frames_<method>.csv from mode_disagreement.py")
    p.add_argument("--n-modes", type=int, default=4, help="fewest modes for (a)")
    p.add_argument("--per-replay", type=int, default=3, help="frames listed per replay for (a)")
    p.add_argument("--min-run", type=int, default=6, help="shortest tie run listed for (b)")
    p.add_argument("--top", type=int, default=15)


def cmd_select_single(args) -> None:
    """Candidates for `single-failure`, from one method's frames CSV.

    (a) frames with at least --n-modes attention modes that the model answered,
        ranked by distance from the pool's median OC1 and median IR - a typical
        multi-mode frame, not the worst one - and listed --per-replay at a time
        so that one replay cannot fill the list;
    (b) the longest runs of consecutive margin-0 frames, with the model's top-2
        flips inside each, next to the fold-wide flip rate on tied frames.
    """
    df = pd.read_csv(args.csv)
    oc = f"OC1@{args.delta}"
    print(f"[select-single] {len(df)} frames in {args.csv}", flush=True)

    pool = df[(df["n_modes"] >= args.n_modes) & (df["n_pred"] > 0)].copy()
    if pool.empty:
        print(f"(a) no answered frame with n_modes >= {args.n_modes}")
    else:
        # OC1 takes only the values k/U, so thousands of frames sit exactly on
        # its median; the IR term breaks that tie towards a typical frame.
        target = pool[oc].median()
        target_ir = pool["IR"].median()
        pool["dist"] = (pool[oc] - target).abs() + (pool["IR"] - target_ir).abs()
        print()
        print(f"(a) pool: {len(pool)} answered frames with n_modes >= {args.n_modes}; "
              f"{oc} mean {pool[oc].mean():.3f}, median {target:.3f}; "
              f"IR mean {pool['IR'].mean():.3f}, median {target_ir:.3f}")
        cols = ["replay", "frame", "progression", "n_modes", "support_top1", "support_top2",
                "IR", oc, "nearest_mode_rank"]
        best = (pool.sort_values(["dist", "replay", "frame"])
                .groupby("replay", sort=False).head(args.per_replay))
        print(best.head(args.top)[[c for c in cols if c in best]].to_string(index=False))

    runs = []
    for replay, g in df.sort_values("frame").groupby("replay"):
        step = g["frame"].diff()
        modal = step.mode().iloc[0] if len(step.mode()) else np.nan
        tie = (g["margin"] == 0).to_numpy()
        adjacent = (step == modal).to_numpy()
        frames = g["frame"].to_numpy()
        flip = g["top2_flip"].to_numpy()
        start = None
        for i in range(len(g) + 1):
            inside = i < len(g) and tie[i] and (start is None or adjacent[i])
            if inside and start is None:
                start = i
            elif not inside and start is not None:
                if i - start >= args.min_run:
                    runs.append({"replay": replay, "start": int(frames[start]),
                                 "end": int(frames[i - 1]), "frames": i - start,
                                 "flips": int(np.nansum(flip[start:i]))})
                start = i if (i < len(g) and tie[i]) else None
    print()
    if not runs:
        print(f"(b) no run of >= {args.min_run} consecutive margin-0 frames")
        return
    rdf = pd.DataFrame(runs)
    rdf["rate"] = rdf["flips"] / rdf["frames"]
    tied = df[df["margin"] == 0]
    print(f"(b) {len(rdf)} runs of >= {args.min_run} margin-0 frames; fold-wide top-2 flip "
          f"rate on margin-0 frames {tied['top2_flip'].mean():.4f}")
    print(rdf.sort_values("frames", ascending=False).head(args.top).to_string(index=False))
    print("  Caption: give the chosen run's flip rate next to the fold-wide one.")


# --------------------------------------------------------------------------

# (subcommand, help, add_args, run), in the order qualitative_figures.py lists them
COMMANDS = [
    ("select", SELECT_HELP, add_select_args, cmd_select),
    ("select-supervision", SELECT_SUPERVISION_HELP, add_select_supervision_args,
     cmd_select_supervision),
    ("select-single", SELECT_SINGLE_HELP, add_select_single_args, cmd_select_single),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = common_parser()
    for name, help_, add_args, run in COMMANDS:
        p = sub.add_parser(name, parents=[common], help=help_)
        add_args(p)
        p.set_defaults(func=run)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
