"""
figures/single_failure.py
=========================

The two failure modes of a single-region observer (thesis chapter on the
single-region baseline; also used in the paper): (a) one multi-mode frame with
the one region the model emits and how many viewers it serves, (b) the region
centre over a run of tied frames against the Top-1 / Top-2 mode tracks. Writes
single_failure_<replay>_<frame>_<tie-replay>_<start>_<end><suffix>.{pdf,png}.

  make figure-fg FIG=single_failure ARGS="--replay 4664 --frame 10331 \
      --tie-replay 1725 --start 11970 --end 12097 --pad 10 \
      --baseline maskrcnn=maskrcnn_win4_vanilla_f1_s456_v6:30@0.5 --label Mask_R-CNN"

Same as `qualitative_figures.py single-failure`.
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
    C_MINOR,
    C_OBS,
    C_TOP1,
    DOUBLE_COL,
    MM,
    SPEC_HELP,
    Source,
    _draw_modes,
    _draw_regions,
    _map_axes,
    _rect,
    _save,
    analyse_method,
    box_center,
    common_parser,
    frame_modes,
    load_frame_npy,
    load_gt,
    minimap_rgb,
    patches,
    plt,
)

HELP = "the two failure modes of a single-region observer"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--replay", required=True, help="replay of the multi-mode frame (a)")
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--tie-replay", required=True, help="replay of the tie run (b)")
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--pad", type=int, default=0, help="frame ids added either side")
    p.add_argument("--axis", choices=["x", "y"], default="x")
    p.add_argument("--baseline", required=True, metavar="SPEC", help=SPEC_HELP)
    p.add_argument("--label", default=None,
                   help="model name in the legend, words joined by '_' (default: the spec's NAME)")
    p.add_argument("--person", choices=["spectator", "observer"], default="spectator",
                   help="what the figure calls a human viewer (thesis: spectator, paper: observer)")
    p.add_argument("--suffix", default="", help="appended to the output file stem")


def run(args) -> None:
    """The two failure modes of a single-region observer, one panel each.

    (a) one multi-mode frame: the observers, their ranked modes and the single
        region the model emits, with how many observers that region serves;
    (b) the region centre over a run of tied frames against the Top-1 and
        Top-2 mode tracks, with the run's top-2 flips.
    Only the model's highest-scoring box is drawn: that is the output of the
    single-region formulation, whatever else the detector proposed.
    """
    # ---- (a)
    replay = str(args.replay)
    gt, height, width, size_hw = load_gt(args, replay)
    if args.frame not in gt:
        raise KeyError(f"frame {args.frame} has no ground truth in replay {replay}")
    obs = gt[args.frame]
    modes = frame_modes(obs, height, width, args)
    bg = minimap_rgb(load_frame_npy(args.input_root, replay, args.frame), fog=not args.no_fog)
    src = Source(args.baseline, args, replay, size_hw)
    boxes, _ = src.get(args.frame)
    boxes = boxes[:1]
    row = analyse_method({args.frame: obs}, {args.frame: (boxes, np.ones(len(boxes)))},
                         height, width, size_hw=size_hw, sigma=args.sigma,
                         min_sep=args.min_sep, rel_threshold=args.rel_threshold,
                         max_modes=args.max_modes, delta=args.delta,
                         straddle_floor=0.15, k_max=1).iloc[0]
    served = row.get(f"OC1@{args.delta}", np.nan)

    # ---- (b)
    t_replay = str(args.tie_replay)
    if t_replay == replay:
        t_gt, t_h, t_w, t_size, t_src = gt, height, width, size_hw, src
    else:
        t_gt, t_h, t_w, t_size = load_gt(args, t_replay)
        t_src = Source(args.baseline, args, t_replay, t_size)
    frames = [f for f in sorted(t_gt) if args.start - args.pad <= f <= args.end + args.pad]
    if len(frames) < 4:
        raise ValueError(f"only {len(frames)} ground-truth frames in the window")
    axis = 1 if args.axis == "x" else 0          # centres are (row, col)
    top1, top2, tie, track = [], [], [], []
    for f in frames:
        m = frame_modes(t_gt[f], t_h, t_w, args)
        top1.append(m.centers[0][axis] if len(m.centers) else np.nan)
        top2.append(m.centers[1][axis] if len(m.centers) > 1 else np.nan)
        tie.append(len(m.support) > 1 and m.support[0] == m.support[1])
        b, _ = t_src.get(f)
        track.append(box_center(b[0])[axis] if len(b) else np.nan)
    one = {}
    for f in frames:
        b, s = t_src.get(f)
        one[f] = (b[:1], s[:1])
    df = analyse_method({f: t_gt[f] for f in frames}, one, t_h, t_w, size_hw=t_size,
                        sigma=args.sigma, min_sep=args.min_sep,
                        rel_threshold=args.rel_threshold, max_modes=args.max_modes,
                        delta=args.delta, straddle_floor=0.15, k_max=1, label="single")
    flips = int(np.nansum(df["top2_flip"]))

    fig = plt.figure(figsize=(DOUBLE_COL, 72 * MM), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.9])
    ax = fig.add_subplot(gs[0])
    ax.imshow(bg, extent=(0, width, height, 0), interpolation="nearest", zorder=0)
    _map_axes(ax, height, width)
    for b in obs:
        _rect(ax, b, C_OBS, lw=0.8, ls=(0, (3, 2)))
    _draw_modes(ax, modes)
    _draw_regions(ax, boxes, 1, C_BASE, C_BASE, number=False)
    u = len(obs)
    ax.set_title(f"(a) {len(modes.centers)} attention modes", loc="left")
    ax.set_xlabel(f"region serves {served * u:.0f} of {u} {args.person}s", fontsize=7)

    ax2 = fig.add_subplot(gs[1])
    fr = np.array(frames)
    step = np.median(np.diff(fr))
    for f, t in zip(fr, tie):
        if t:
            ax2.axvspan(f - step / 2, f + step / 2, color="#ece6f4", lw=0, zorder=0)
    ax2.scatter(fr, top1, s=6, color=C_TOP1, alpha=0.45, lw=0, zorder=1)
    ax2.scatter(fr, top2, s=6, color=C_MINOR, alpha=0.45, lw=0, zorder=1)
    ax2.plot(fr, track, color=C_BASE, lw=1.4, zorder=3)
    ax2.set_xlim(fr[0], fr[-1])
    ax2.set_ylim(0, t_w if axis == 1 else t_h)
    ax2.set_xlabel("frame")
    ax2.set_ylabel(f"region centre {args.axis} (tiles)")
    ax2.grid(True, lw=0.4, alpha=0.4)
    ax2.set_title(f"(b) tied frames shaded; {flips} top-2 flips over {len(frames)} frames",
                  loc="left")

    handles = [
        patches.Patch(fill=False, edgecolor=C_OBS, linestyle="--", label=f"{args.person} viewport"),
        plt.Line2D([], [], marker="o", ls="", color=C_TOP1, label="Top-1 mode"),
        plt.Line2D([], [], marker="o", ls="", color=C_MINOR, label="minority / Top-2 mode"),
        plt.Line2D([], [], color=C_BASE, lw=1.4,
                   label=f"{(args.label or src.name).replace('_', ' ')} region"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    print(f"[single-failure] (a) replay {replay} frame {args.frame}: {len(modes.centers)} "
          f"modes, support {modes.support.tolist()}, serves {served * u:.0f}/{u}; "
          f"(b) replay {t_replay} frames {frames[0]}-{frames[-1]} ({len(frames)}): "
          f"{sum(tie)} tied, {flips} top-2 flips", flush=True)
    _save(fig, args.outdir,
          f"single_failure_{replay}_{args.frame}_{t_replay}_{args.start}_{args.end}{args.suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, parents=[common_parser()],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
