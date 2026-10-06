"""
figures/trajectory.py
=====================

Qualitative figure (3): the primary-region center over a window of
consecutive frames for both methods, with the Top-1 / Top-2 mode tracks, the
tie frames (support margin 0) shaded and each method's top-2 flips in the
window. Writes qual3_trajectory_<replay>_<start>_<end>.{pdf,png}.

  make figure-fg FIG=trajectory ARGS="--replay 3613 --start 1395 --end 1445 \
      --baseline maskrcnn=maskrcnn_win4_vanilla_f1_s456_v6:30@0.5 \
      --director director=dc_full_b16_f1_s456_v6:30"

Same as `qualitative_figures.py trajectory`.
"""

from __future__ import annotations

import argparse
import os
import sys

_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import numpy as np

from figures.common import (
    C_BASE,
    C_MAIN,
    C_MINOR,
    C_PIP,
    C_TOP1,
    DOUBLE_COL,
    MM,
    SPEC_HELP,
    Source,
    _save,
    analyse_method,
    box_center,
    common_parser,
    display_name,
    legend_ncol,
    frame_modes,
    load_gt,
    plt,
)

HELP = "figure (3)"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--replay", required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--pad", type=int, default=0, help="frame ids added either side")
    p.add_argument("--axis", choices=["x", "y"], default="x")
    p.add_argument("--baseline", required=True, metavar="SPEC", help=SPEC_HELP)
    p.add_argument("--director", required=True, metavar="SPEC", help=SPEC_HELP)


def run(args) -> None:
    replay = str(args.replay)
    gt, height, width, size_hw = load_gt(args, replay)
    frames = [f for f in sorted(gt) if args.start - args.pad <= f <= args.end + args.pad]
    if len(frames) < 4:
        raise ValueError(f"only {len(frames)} ground-truth frames in the window")
    base = Source(args.baseline, args, replay, size_hw)
    dire = Source(args.director, args, replay, size_hw)
    axis = 1 if args.axis == "x" else 0          # centers are (row, col)

    top1, top2, tie = [], [], []
    for f in frames:
        m = frame_modes(gt[f], height, width, args)
        top1.append(m.centers[0][axis] if len(m.centers) else np.nan)
        top2.append(m.centers[1][axis] if len(m.centers) > 1 else np.nan)
        tie.append(len(m.support) > 1 and m.support[0] == m.support[1])

    def primary(src):
        out = []
        for f in frames:
            b, _ = src.get(f)
            out.append(box_center(b[0])[axis] if len(b) else np.nan)
        return np.array(out)

    b_track, d_track = primary(base), primary(dire)
    pip_f, pip_v = [], []
    for f in frames:
        b, _ = dire.get(f)
        for box in b[1:args.k]:
            pip_f.append(f)
            pip_v.append(box_center(box)[axis])

    # flip counts in this window, by the same code as the paper's tables
    flips = {}
    sub_gt = {f: gt[f] for f in frames}
    for name, src in (("base", base), ("dir", dire)):
        df = analyse_method(sub_gt, {f: src.get(f) for f in frames}, height, width,
                            size_hw=size_hw, sigma=args.sigma, min_sep=args.min_sep,
                            rel_threshold=args.rel_threshold, max_modes=args.max_modes,
                            delta=args.delta, straddle_floor=0.15, k_max=args.k,
                            label=name)
        flips[name] = int(np.nansum(df["top2_flip"]))

    fr = np.array(frames)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 62 * MM))
    step = np.median(np.diff(fr))
    for f, t in zip(fr, tie):
        if t:
            ax.axvspan(f - step / 2, f + step / 2, color="#ece6f4", lw=0, zorder=0)
    ax.scatter(fr, top1, s=6, color=C_TOP1, alpha=0.35, lw=0, zorder=1, label="Top-1 mode")
    ax.scatter(fr, top2, s=6, color=C_MINOR, alpha=0.35, lw=0, zorder=1, label="Top-2 mode")
    # Drawn wider than the Director track and under it, so that a window where
    # the two primaries coincide - which is the common case, since both are
    # anchored on the Top-1 mode - reads as one line inside another rather than
    # as a missing baseline.
    ax.plot(fr, b_track, color=C_BASE, lw=3.5, alpha=0.45, solid_capstyle="butt",
            zorder=3, label=f"{display_name(base.name)} top-1")
    if pip_f:
        ax.scatter(pip_f, pip_v, s=5, marker="s", color=C_PIP, lw=0, zorder=3,
                   label="Director-CenterNet auxiliary")
    ax.plot(fr, d_track, color=C_MAIN, lw=1.5, zorder=4, label="Director-CenterNet primary")
    ax.set_xlim(fr[0], fr[-1])
    ax.set_ylim(0, width if axis == 1 else height)
    ax.set_xlabel("frame")
    ax.set_ylabel(f"region center {args.axis} (tiles)")
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.legend(loc="upper center", ncol=legend_ncol(5), frameon=False, bbox_to_anchor=(0.5, -0.2))
    ax.text(1, 1.02, f"shaded: support margin 0 · top-2 flips {display_name(base.name)} {flips['base']}, "
            f"Director-CenterNet {flips['dir']}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color="#5b6474")
    print(f"[trajectory] replay {replay} frames {frames[0]}-{frames[-1]} ({len(frames)}): "
          f"{sum(tie)} tie frames; top-2 flips {base.name} {flips['base']}, "
          f"Director {flips['dir']}", flush=True)
    _save(fig, args.outdir, f"qual3_trajectory_{replay}_{args.start}_{args.end}{args.suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, parents=[common_parser()],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
