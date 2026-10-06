"""
figures/supervision_grid.py
===========================

Several frames through the supervision construction, one row per frame:
observer viewports, coverage a_t, ranked modes, rendered target Y_t. Writes
qual6_supervision_grid_<rows>.{pdf,png}. Frames come as REPLAY:FRAME pairs, in
the order they should appear.

  make figure-fg FIG=supervision_grid ARGS="--frames 3613:340 4664:17160 1725:6020 275:44960"

Same as `qualitative_figures.py supervision-grid`.
"""

from __future__ import annotations

import argparse
import os
import sys

_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import numpy as np

import config
from figures.common import (
    C_MINOR,
    C_OBS,
    C_TOP1,
    DOUBLE_COL,
    _map_axes,
    _rect,
    _save,
    common_parser,
    coverage_maps,
    frame_modes,
    load_frame_npy,
    load_gt,
    minimap_rgb,
    plt,
)

HELP = "several frames through the supervision construction"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--frames", nargs="+", required=True, metavar="REPLAY:FRAME")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--render-sigma", type=float, default=config.DIRECTOR_RENDER_SIGMA)


def run(args) -> None:
    """Several frames through the supervision construction, one per row.

    The same four panels as `supervision`, stacked, so that the construction is
    read as a procedure applied to different frames rather than as one picture.
    Column headings appear once; each row is labelled by what distinguishes it,
    which is the support vector and how far apart the modes fell.

    Takes REPLAY:FRAME pairs, in the order they should appear.
    """
    import torch
    from losses.director_losses import render_gaussian_heatmap_targets

    pairs = []
    for spec in args.frames:
        r, _, f = spec.partition(":")
        if not f:
            raise ValueError(f"--frames wants REPLAY:FRAME, got {spec!r}")
        pairs.append((str(r), int(f)))

    ratios = [0.30] + [1.0] * 4
    nrow = len(pairs)
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL / sum(ratios) * nrow + 0.75),
                     layout="constrained")
    gs = fig.add_gridspec(nrow, len(ratios), width_ratios=ratios)
    titles = [f"{args.person} viewports", r"coverage $a_t$",
              r"ranked modes $\mathcal{M}_t$", r"target $\mathcal{Y}_t$"]

    for row, (replay, frame) in enumerate(pairs):
        gt, height, width, size_hw = load_gt(args, replay)
        if frame not in gt:
            raise KeyError(f"frame {frame} has no ground truth in replay {replay}")
        obs = gt[frame]
        modes = frame_modes(obs, height, width, args)
        cov, smooth = coverage_maps(obs, height, width, args.sigma)
        bg = minimap_rgb(load_frame_npy(args.input_root, replay, frame),
                         fog=not args.no_fog)
        stride = args.stride
        rendered = render_gaussian_heatmap_targets(
            modes_list=[{"centers": torch.as_tensor(modes.centers, dtype=torch.float32),
                         "support": torch.as_tensor(modes.support, dtype=torch.float32)}],
            batch_size=1, feat_h=height // stride, feat_w=width // stride,
            stride=stride, device=torch.device("cpu"), render_sigma=args.render_sigma,
            u_observers=len(obs), viewport_size_hw=(int(size_hw[0]), int(size_hw[1])))
        target = rendered["Y_all"][0, 0].numpy()
        ext = (0, width, height, 0)

        sup = [int(s) for s in modes.support]
        gap = min(float(np.linalg.norm(modes.centers[i] - modes.centers[j]))
                  for i in range(len(sup)) for j in range(i + 1, len(sup)))
        lab = fig.add_subplot(gs[row, 0])
        lab.axis("off")
        lab.text(0.95, 0.5, f"$n = {', '.join(str(s) for s in sup)}$\n"
                            f"gap {gap:.0f} tiles",
                 fontsize=6, color="#4a525c", ha="right", va="center",
                 transform=lab.transAxes, linespacing=1.5)

        a = [fig.add_subplot(gs[row, c]) for c in range(1, 5)]
        a[0].imshow(bg, extent=ext, interpolation="nearest")
        for b in obs:
            _rect(a[0], b, C_OBS, lw=0.8, ls=(0, (3, 2)))
        a[1].imshow(cov, extent=ext, cmap="Greys", vmin=0,
                    vmax=max(cov.max(), 1e-9), interpolation="nearest")
        a[2].imshow(smooth / max(smooth.max(), 1e-9), extent=ext, cmap="viridis",
                    vmin=0, vmax=1, interpolation="nearest")
        for i, c in enumerate(modes.centers):
            a[2].scatter([c[1]], [c[0]], s=14,
                         color=C_TOP1 if i == 0 else C_MINOR,
                         edgecolor="white", linewidth=0.6, zorder=5)
        a[3].imshow(target, extent=ext, cmap="viridis", vmin=0, vmax=1,
                    interpolation="nearest")
        for x in a:
            _map_axes(x, height, width)
        if row == 0:
            for x, t in zip(a, titles):
                x.set_title(t, loc="left", fontsize=6.5)
        print(f"[supervision-grid] {replay}/{frame}: support {sup}, gap {gap:.0f}",
              flush=True)

    # No equation number in the image: matplotlib renders the "~" of "Eq.~(13)"
    # literally, and a number baked into a figure goes stale when the paper
    # renumbers. The caption carries the reference.
    fig.supxlabel(
        r"Modes are ranked by support $n$ and rendered at amplitude $1$ for the "
        r"Top-1 mode and $n/U$ for the rest.",
        fontsize=6.5)
    _save(fig, args.outdir, f"qual6_supervision_grid_{nrow}{args.suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, parents=[common_parser()],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
