"""
figures/teaser.py
=================

The opening figure: one real frame, its observers and their ranked modes, one
column wide. Writes qual0_teaser_<replay>_<frame>.{pdf,png} to --outdir.

  make figure-fg FIG=teaser ARGS="--replay 4520 --frame 10298"

Same as `qualitative_figures.py teaser`.
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
    C_MINOR,
    C_OBS,
    C_TOP1,
    SINGLE_COL,
    _draw_modes,
    _map_axes,
    _rect,
    _save,
    common_parser,
    frame_modes,
    load_frame_npy,
    load_gt,
    minimap_rgb,
    patches,
    plt,
)

HELP = "the opening figure: observers and ranked modes, one column wide"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--replay", required=True)
    p.add_argument("--frame", type=int, required=True)


def run(args) -> None:
    """The opening figure: one real frame, its observers, and its ranked modes.

    Panel (a) of `compare`, on its own and at one column rather than two. It
    states the problem the paper is about - the observers do not agree, and
    their disagreement has a ranking - without showing any model output, so
    nothing here has to be qualified by how well a method happened to do on the
    frame.
    """
    replay = str(args.replay)
    gt, height, width, size_hw = load_gt(args, replay)
    if args.frame not in gt:
        raise KeyError(f"frame {args.frame} has no ground truth in replay {replay}")
    obs = gt[args.frame]
    modes = frame_modes(obs, height, width, args)
    bg = minimap_rgb(load_frame_npy(args.input_root, replay, args.frame), fog=not args.no_fog)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL + 0.30), layout="constrained")
    ax.imshow(bg, extent=(0, width, height, 0), interpolation="nearest", zorder=0)
    _map_axes(ax, height, width)
    for b in obs:
        _rect(ax, b, C_OBS, lw=0.9, ls=(0, (3, 2)))
    _draw_modes(ax, modes)

    handles = [
        patches.Patch(fill=False, edgecolor=C_OBS, linestyle="--", label=f"{args.person} viewport"),
        plt.Line2D([], [], marker="o", ls="", color=C_TOP1, label="Top-1 mode"),
        plt.Line2D([], [], marker="o", ls="", color=C_MINOR, label="minority mode"),
    ]
    # One row: a legend of three entries at ncol=2 fills column-major and leaves
    # the third entry alone on a second row.
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False,
               fontsize=6.5, handletextpad=0.4, columnspacing=1.0)

    sep = [float(np.linalg.norm(modes.centers[i] - modes.centers[j]))
           for i in range(len(modes.centers)) for j in range(i + 1, len(modes.centers))]
    print(f"[teaser] replay {replay} frame {args.frame}: {len(modes.centers)} modes, "
          f"support {modes.support.tolist()}, min separation "
          f"{min(sep) if sep else float('nan'):.0f} tiles", flush=True)
    _save(fig, args.outdir, f"qual0_teaser_{replay}_{args.frame}{args.suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, parents=[common_parser()],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
