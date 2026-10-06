"""
figures/supervision.py
======================

The supervision row of the architecture figure, from one real frame: observer
viewports -> coverage a_t -> smoothed field with the ranked modes -> rendered
target Y_t. Writes qual4_supervision_<replay>_<frame>.{pdf,png}.

  make figure-fg FIG=supervision ARGS="--replay 4520 --frame 10298"

Same as `qualitative_figures.py supervision`.
"""

from __future__ import annotations

import argparse
import os
import sys

_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import config
from figures.common import (
    C_MAIN,
    C_MINOR,
    C_OBS,
    C_PIP,
    C_TOP1,
    DOUBLE_COL,
    _label,
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

HELP = "the supervision row of the architecture figure"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--replay", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--stride", type=int, default=4,
                   help="output stride the target is rendered at")
    p.add_argument("--render-sigma", type=float, default=config.DIRECTOR_RENDER_SIGMA,
                   help="Gaussian sigma of the rendered target, in tiles")


def run(args) -> None:
    """The supervision row of the architecture figure, from a real frame.

    Observer viewports -> coverage a_t -> its smoothed field with the ranked
    modes -> the rendered target Y_t. This is the left-to-right pipeline of
    Section "Ranked Modes of the Observer Distribution", and every panel is
    computed by the same code the training run uses, so the amplitudes in the
    last panel are the ones the loss actually sees.

    It is a row rather than a whole figure because the other half of the
    architecture diagram - the network schematic - contains no data and is
    drawn by hand. Rendering this half here is what keeps the heatmaps from
    being invented.
    """
    import torch
    from losses.director_losses import render_gaussian_heatmap_targets

    replay = str(args.replay)
    gt, height, width, size_hw = load_gt(args, replay)
    if args.frame not in gt:
        raise KeyError(f"frame {args.frame} has no ground truth in replay {replay}")
    obs = gt[args.frame]
    modes = frame_modes(obs, height, width, args)
    cov, smooth = coverage_maps(obs, height, width, args.sigma)
    bg = minimap_rgb(load_frame_npy(args.input_root, replay, args.frame), fog=not args.no_fog)

    stride = args.stride
    feat_h, feat_w = height // stride, width // stride
    rendered = render_gaussian_heatmap_targets(
        modes_list=[{"centers": torch.as_tensor(modes.centers, dtype=torch.float32),
                     "support": torch.as_tensor(modes.support, dtype=torch.float32)}],
        batch_size=1, feat_h=feat_h, feat_w=feat_w, stride=stride,
        device=torch.device("cpu"), render_sigma=args.render_sigma,
        u_observers=len(obs), viewport_size_hw=(int(size_hw[0]), int(size_hw[1])),
    )
    target = rendered["Y_all"][0, 0].numpy()

    # Four panels with an arrow between each pair. The arrows get gridspec
    # columns of their own so their position does not depend on where the
    # layout engine happens to put the axes.
    ratios = [1, 0.18, 1, 0.18, 1, 0.18, 1]
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL / sum(ratios) + 0.55),
                     layout="constrained")
    gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios)
    ax = [fig.add_subplot(gs[0, i]) for i in (0, 2, 4, 6)]
    ext = (0, width, height, 0)

    ax[0].imshow(bg, extent=ext, interpolation="nearest")
    for b in obs:
        _rect(ax[0], b, C_OBS, lw=0.9, ls=(0, (3, 2)))
    ax[0].set_title(f"{args.person} viewports, $U = {len(obs)}$", loc="left", fontsize=7)

    # Scaled to the frame's own maximum, not to U: on a frame where no more
    # than three of the five overlap, a 0..U scale renders the panel blank and
    # hides the stacking it exists to show.
    # `coverage_maps` divides by U, so the darkest cell here is the fraction of
    # observers whose viewport contains it, and the panel is scaled to the
    # frame's own maximum: on a frame where no more than three of the five
    # overlap, a 0..1 scale renders it blank and hides the stacking it exists
    # to show.
    ax[1].imshow(cov, extent=ext, cmap="Greys", vmin=0, vmax=max(cov.max(), 1e-9),
                 interpolation="nearest")
    ax[1].set_title(rf"coverage $a_t$, max ${round(cov.max() * len(obs))}/{len(obs)}$",
                    loc="left", fontsize=7)

    def place(a, y, x, text, color):
        """Label beside a mode without running into the title or the next label."""
        dy = 7 if y < height * 0.18 else -6          # below if near the top edge
        xx = min(max(x, width * 0.08), width * 0.92)
        _label(a, xx, y + dy, text, color, 6.5)

    ax[2].imshow(smooth / max(smooth.max(), 1e-9), extent=ext, cmap="viridis",
                 vmin=0, vmax=1, interpolation="nearest")
    for i, (c, n) in enumerate(zip(modes.centers, modes.support)):
        col = C_TOP1 if i == 0 else C_MINOR
        ax[2].scatter([c[1]], [c[0]], s=18, color=col, edgecolor="white",
                      linewidth=0.7, zorder=5)
        # Rank only. Two single-observer modes can sit close enough that
        # "#2 n=1" and "#3 n=1" collide; the supports are in the caption below,
        # in rank order, which is the same information without the overlap.
        place(ax[2], c[0], c[1], f"#{i + 1}", col)
    ax[2].set_title(r"ranked modes $\mathcal{M}_t$", loc="left", fontsize=7)

    ax[3].imshow(target, extent=ext, cmap="viridis", vmin=0, vmax=1,
                 interpolation="nearest")
    for i, (c, n) in enumerate(zip(modes.centers, modes.support)):
        amp = 1.0 if i == 0 else float(n) / len(obs)
        place(ax[3], c[0], c[1], f"{amp:.1f}", C_MAIN if i == 0 else C_PIP)
    ax[3].set_title(rf"target $\mathcal{{Y}}_t$, {feat_h}$\times${feat_w}",
                    loc="left", fontsize=7)

    for a in ax:
        _map_axes(a, height, width)

    for i in (1, 3, 5):
        a = fig.add_subplot(gs[0, i])
        a.axis("off")
        a.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5), xycoords="axes fraction",
                   arrowprops=dict(arrowstyle="-|>", color="#39414f", lw=1.0))

    fig.supxlabel(
        rf"support $n = {', '.join(str(int(s)) for s in modes.support)}$;  "
        rf"amplitude $1$ for the Top-1 mode and $n/U$ for the rest",
        fontsize=6.5)

    print(f"[supervision] replay {replay} frame {args.frame}: "
          f"{len(modes.centers)} modes, support {modes.support.tolist()}, "
          f"target max {target.max():.3f}", flush=True)
    _save(fig, args.outdir, f"qual4_supervision_{replay}_{args.frame}{args.suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, parents=[common_parser()],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
