"""
figures/heatmap.py
==================

Qualitative figure (2): one frame as four panels - coverage a_t, the smoothed
coverage with its ranked modes, the predicted heatmap Y_hat_t and the decoded
regions over the minimap. The heatmap is not saved by inference, so the
Director checkpoint is re-run on the frame and its decoded scores are checked
against the saved predictions first; it wants the GPU. Writes
qual2_heatmap_<replay>_<frame>.{pdf,png}.

  make figure-fg FIG=heatmap ARGS="--replay 4520 --frame 10298 \
      --director director=dc_full_b16_f1_s456_v6:30"

Same as `qualitative_figures.py heatmap`.
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
    C_MAIN,
    C_PIP,
    DOUBLE_COL,
    SPEC_HELP,
    Source,
    _director_heatmap,
    _draw_modes,
    _draw_regions,
    _label,
    _map_axes,
    _rect,
    _save,
    box_center,
    common_parser,
    coverage_maps,
    frame_modes,
    load_frame_npy,
    load_gt,
    minimap_rgb,
    plt,
)

HELP = "figure (2)"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--replay", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--director", required=True, metavar="SPEC", help=SPEC_HELP)
    p.add_argument("--window-size", type=int, default=4,
                   help="used only if inference_provenance.json does not record it")
    p.add_argument("--include-components", nargs="+",
                   default=["worker", "ground", "air", "building", "vision"],
                   help="used only if inference_provenance.json does not record it")
    p.add_argument("--conf-threshold", type=float, default=0.1,
                   help="tau; used only if inference_provenance.json does not record it")


def run(args) -> None:
    replay = str(args.replay)
    gt, height, width, size_hw = load_gt(args, replay)
    obs = gt[args.frame]
    modes = frame_modes(obs, height, width, args)
    cov, smooth = coverage_maps(obs, height, width, args.sigma)
    bg = minimap_rgb(load_frame_npy(args.input_root, replay, args.frame), fog=not args.no_fog)
    dire = Source(args.director, args, replay, size_hw)
    saved_boxes, saved_scores = dire.get(args.frame)

    hm, _xyxy, scores, stride, tau = _director_heatmap(args, dire, replay, args.frame)

    # The panel is only honest if the forward pass reproduces what was scored.
    n = min(len(scores), len(saved_scores))
    if len(scores) != len(saved_scores) or not np.allclose(scores[:n], saved_scores[:n], atol=1e-3):
        print(f"[heatmap] WARNING: re-run scores {np.round(scores, 3).tolist()} differ from "
              f"saved {np.round(saved_scores, 3).tolist()}; check checkpoint / threshold",
              flush=True)

    # The colorbar gets a row of its own rather than `colorbar(ax=axes[1:3])`,
    # which takes its space *out of* those two axes. They carry imshow and so
    # have a fixed aspect: losing height loses width with it, and (b) and (c)
    # came out smaller than (a) and (d) and centred at a different height.
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL / 4 + 0.75), layout="constrained")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.05])
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ext = (0, width, height, 0)
    axes[0].imshow(cov, extent=ext, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    for b in obs:
        _rect(axes[0], b, C_MAIN, lw=0.6, alpha=0.8)
    axes[0].set_title(r"(a) Coverage $a_t$", loc="left")

    im = axes[1].imshow(smooth / max(smooth.max(), 1e-9), extent=ext, cmap="viridis",
                        vmin=0, vmax=1, interpolation="nearest")
    _draw_modes(axes[1], modes)
    # Kept short: the panels are square images in wider slots, so a left-aligned
    # title longer than "(d) Decoded regions" runs into the next panel's.
    axes[1].set_title(r"(b) Modes of $\tilde{a}_t$", loc="left")

    axes[2].imshow(hm, extent=ext, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    for r, b in enumerate(saved_boxes):
        c = box_center(b)
        _label(axes[2], c[1], c[0] - 5, f"{saved_scores[r]:.2f}", C_MAIN if r == 0 else C_PIP, 6.5)
    axes[2].set_title(r"(c) Predicted $\hat{\mathcal{Y}}_t$", loc="left")

    axes[3].imshow(bg, extent=ext, interpolation="nearest")
    _draw_regions(axes[3], saved_boxes, args.k, C_MAIN, C_PIP)
    axes[3].set_title("(d) Decoded regions", loc="left")

    for ax in axes:
        _map_axes(ax, height, width)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[1, 1:3]), orientation="horizontal")
    cb.set_ticks([0, 0.2, 0.5, 1])
    cb.ax.tick_params(labelsize=6.5)
    cb.set_label(f"(b) normalized to its maximum; (c) sigmoid score, 1/U = {1 / len(obs):.1f}",
                 fontsize=6.5)
    axes[3].set_xlabel(rf"$\tau$ = {tau:g}, stride {stride}", fontsize=6.5)
    print(f"[heatmap] replay {replay} frame {args.frame}: modes support "
          f"{modes.support.tolist()}, peak scores {np.round(saved_scores, 3).tolist()}", flush=True)
    _save(fig, args.outdir, f"qual2_heatmap_{replay}_{args.frame}{args.suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, parents=[common_parser()],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
