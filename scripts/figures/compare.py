"""
figures/compare.py
==================

Qualitative figure (1): one frame, three panels - the U observers and their
ranked modes, the baseline's top-K boxes, Director-CenterNet's primary and
auxiliary regions - with each method's per-frame OC_K@delta. Writes
qual1_compare_<replay>_<frame>.{pdf,png}.

  make figure-fg FIG=compare ARGS="--replay 4520 --frame 10298 \
      --baseline maskrcnn=maskrcnn_win4_vanilla_f1_s456_v6:30@0.5 \
      --director director=dc_full_b16_f1_s456_v6:30"

Same as `qualitative_figures.py compare`.
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
    C_OBS,
    C_PIP,
    C_TOP1,
    DOUBLE_COL,
    SPEC_HELP,
    Source,
    _draw_modes,
    _draw_regions,
    _map_axes,
    _rect,
    _save,
    analyse_method,
    common_parser,
    display_name,
    frame_modes,
    load_frame_npy,
    load_gt,
    minimap_rgb,
    patches,
    plt,
)

HELP = "figure (1)"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--replay", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--baseline", required=True, metavar="SPEC", help=SPEC_HELP)
    p.add_argument("--director", required=True, metavar="SPEC", help=SPEC_HELP)


def run(args) -> None:
    replay = str(args.replay)
    gt, height, width, size_hw = load_gt(args, replay)
    if args.frame not in gt:
        raise KeyError(f"frame {args.frame} has no ground truth in replay {replay}")
    obs = gt[args.frame]
    modes = frame_modes(obs, height, width, args)
    bg = minimap_rgb(load_frame_npy(args.input_root, replay, args.frame), fog=not args.no_fog)
    base = Source(args.baseline, args, replay, size_hw)
    dire = Source(args.director, args, replay, size_hw)
    b_boxes, _ = base.get(args.frame)
    d_boxes, _ = dire.get(args.frame)

    # the per-frame number the caption quotes, computed by the paper's own code
    oc = {}
    for src, boxes in (("base", b_boxes), ("dir", d_boxes)):
        row = analyse_method({args.frame: obs}, {args.frame: (boxes, np.ones(len(boxes)))},
                             height, width, size_hw=size_hw, sigma=args.sigma,
                             min_sep=args.min_sep, rel_threshold=args.rel_threshold,
                             max_modes=args.max_modes, delta=args.delta,
                             straddle_floor=0.15, k_max=args.k).iloc[0]
        oc[src] = row.get(f"OC{args.k}@{args.delta}", np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL / 3 + 0.55),
                             layout="constrained")
    titles = [f"(a) Human {args.person}s, U = {len(obs)}",
              f"(b) {display_name(base.name)}, top-{args.k}",
              "(c) Director-CenterNet"]
    for ax, title in zip(axes, titles):
        ax.imshow(bg, extent=(0, width, height, 0), interpolation="nearest", zorder=0)
        _map_axes(ax, height, width)
        ax.set_title(title, loc="left")

    for b in obs:
        _rect(axes[0], b, C_OBS, lw=0.9, ls=(0, (3, 2)))
    _draw_modes(axes[0], modes)

    for ax in axes[1:]:
        for b in obs:
            _rect(ax, b, C_OBS, lw=0.6, ls=(0, (3, 2)), alpha=0.5, z=2)
    _draw_regions(axes[1], b_boxes, args.k, C_BASE, C_BASE)
    _draw_regions(axes[2], d_boxes, args.k, C_MAIN, C_PIP)

    u = len(obs)
    axes[1].set_xlabel(f"OC{args.k}@{args.delta} = {oc['base'] * u:.0f}/{u}", fontsize=7)
    axes[2].set_xlabel(f"OC{args.k}@{args.delta} = {oc['dir'] * u:.0f}/{u}", fontsize=7)

    handles = [
        patches.Patch(fill=False, edgecolor=C_OBS, linestyle="--", label=f"{args.person} viewport"),
        plt.Line2D([], [], marker="o", ls="", color=C_TOP1, label="Top-1 mode"),
        plt.Line2D([], [], marker="o", ls="", color=C_MINOR, label="minority mode"),
        patches.Patch(fill=False, edgecolor=C_BASE, label=f"{display_name(base.name)} box"),
        patches.Patch(fill=False, edgecolor=C_MAIN, label="primary"),
        patches.Patch(fill=False, edgecolor=C_PIP, label="auxiliary"),
    ]
    # "outside" so the constrained layout reserves the strip for it. Anchoring
    # it below the canvas instead only worked while _save cropped with
    # bbox_inches="tight", which grew the canvas to take the legend in; without
    # that it landed on top of the OC labels.
    fig.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False)
    print(f"[compare] replay {replay} frame {args.frame}: {len(modes.centers)} modes, "
          f"support {modes.support.tolist()}; {base.name} {len(b_boxes)} boxes, "
          f"Director {len(d_boxes)} regions; OC{args.k}@{args.delta} "
          f"{oc['base']:.2f} vs {oc['dir']:.2f}", flush=True)
    _save(fig, args.outdir, f"qual1_compare_{replay}_{args.frame}{args.suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, parents=[common_parser()],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
