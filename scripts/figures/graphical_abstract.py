"""
figures/graphical_abstract.py
=============================

The journal's graphical abstract, 13 x 5 cm, from one real frame: stacked
game state, Director-CenterNet's center heatmap (re-run from the checkpoint),
the ranked regions and the crops they select. Writes
graphical_abstract_<replay>_<frame>.{pdf,png}. Re-runs a checkpoint, so it
wants the GPU (it falls back to the CPU).

  make figure-fg FIG=graphical_abstract ARGS="--replay 4520 --frame 10298 \
      --director director=dc_full_b16_f1_s456_v6:30"

Same as `qualitative_figures.py graphical-abstract`.
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
    MM,
    SPEC_HELP,
    Source,
    _director_heatmap,
    _draw_regions,
    _label,
    _map_axes,
    _save,
    box_center,
    common_parser,
    load_frame_npy,
    load_gt,
    minimap_rgb,
    patches,
    path_effects,
    plt,
)

HELP = "the journal's graphical abstract, 13 x 5 cm"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--replay", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--director", required=True, metavar="SPEC", help=SPEC_HELP)
    p.add_argument("--stack-delta", type=int, default=8,
                   help="temporal stride of the stack (--delta is the overlap threshold)")
    p.add_argument("--stack", type=int, default=3, help="past frames in the stack")
    p.add_argument("--window-size", type=int, default=4)
    p.add_argument("--include-components", nargs="+",
                   default=["worker", "ground", "air", "building", "vision"])
    p.add_argument("--conf-threshold", type=float, default=0.1)


def run(args) -> None:
    """The journal's graphical abstract, from one real frame.

    Elsevier asks for an image readable at 13 x 5 cm, which is a 2.6:1
    landscape and nothing like the figures in the paper. The size is fixed
    here rather than left to whoever exports it, because a hand-drawn version
    scaled to that box is how the label sizes drift.

    Four stages: the stacked game-state tensor, the heatmap the checkpoint
    produces from it, the ranked region set decoded from that heatmap, and the
    crops those regions select. The crops are the real ones - a region is
    12 x 20 tiles, so they are coarse, and that is what the input resolution
    is. The alternative was to draw a game screenshot we do not have.
    """
    replay = str(args.replay)
    gt, height, width, size_hw = load_gt(args, replay)
    obs = gt[args.frame]
    dire = Source(args.director, args, replay, size_hw)
    boxes, scores = dire.get(args.frame)
    hm, _xyxy, re_scores, stride, tau = _director_heatmap(args, dire, replay, args.frame)
    bg = minimap_rgb(load_frame_npy(args.input_root, replay, args.frame), fog=not args.no_fog)

    # The stack the network actually reads: L past frames at spacing delta,
    # oldest behind.
    hist = []
    for j in range(args.stack, -1, -1):
        f = args.frame - j * args.stack_delta
        try:
            hist.append(minimap_rgb(load_frame_npy(args.input_root, replay, f),
                                    fog=not args.no_fog))
        except FileNotFoundError:
            pass

    GA_W, GA_H = 130 * MM, 50 * MM
    ratios = [1, 0.2, 1, 0.2, 1, 0.2, 0.8]
    fig = plt.figure(figsize=(GA_W, GA_H), layout="constrained")
    gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios)
    ext = (0, width, height, 0)

    # (1) game-state stack
    a0 = fig.add_subplot(gs[0, 0])
    off = width * 0.055
    for i, im in enumerate(hist):
        d = (len(hist) - 1 - i) * off
        a0.imshow(im, extent=(d, width + d, height - d, -d), zorder=i,
                  interpolation="nearest", alpha=1.0 if i == len(hist) - 1 else 0.92)
        a0.add_patch(patches.Rectangle((d, -d), width, height, fill=False,
                                       edgecolor="#9aa3ad", lw=0.5, zorder=i))
    a0.set_xlim(-off * 0.4, width + off * (len(hist) - 0.2))
    a0.set_ylim(height + off * 0.4, -off * (len(hist) - 0.2))
    a0.set_aspect("equal"); a0.set_xticks([]); a0.set_yticks([])
    for s in a0.spines.values():
        s.set_visible(False)
    # Titles are two short lines each. A column here is about 3 cm wide, which
    # at 7 pt holds roughly twenty characters; anything longer runs into the
    # next column, and the layout engine will not stop it.
    a0.set_title(f"game state\n{len(hist)} frames", loc="left", fontsize=7)

    # (2) predicted heatmap
    a1 = fig.add_subplot(gs[0, 2])
    a1.imshow(hm, extent=ext, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    for r, b in enumerate(boxes[:args.k]):
        c = box_center(b)
        _label(a1, c[1], c[0] - 7, str(r + 1), C_MAIN if r == 0 else C_PIP, 7)
    _map_axes(a1, height, width)
    a1.set_title("Director-CenterNet\ncenter heatmap", loc="left", fontsize=7)

    # (3) the decoded region set
    a2 = fig.add_subplot(gs[0, 4])
    a2.imshow(bg, extent=ext, interpolation="nearest")
    _draw_regions(a2, boxes, args.k, C_MAIN, C_PIP)
    _map_axes(a2, height, width)
    # No "--": matplotlib is not LaTeX and renders it as two hyphens.
    a2.set_title("ranked regions\n1 primary, 0 to 2 aux", loc="left", fontsize=7)

    # (4) what each region selects
    sub = gs[0, 6].subgridspec(len(boxes[:args.k]), 1, hspace=0.35)
    for r, b in enumerate(boxes[:args.k]):
        x, y, w, h = [int(round(v)) for v in b]
        x0, y0 = max(0, x), max(0, y)
        crop = bg[y0:min(height, y0 + h), x0:min(width, x0 + w)]
        a = fig.add_subplot(sub[r])
        a.imshow(crop, interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        col = C_MAIN if r == 0 else C_PIP
        for s in a.spines.values():
            s.set_color(col); s.set_linewidth(1.4)
        # Inside the crop: a label beside it would have to fit in the gutter
        # between two columns, which at this width there is not.
        t = a.text(0.04, 0.88, "primary" if r == 0 else f"aux {r + 1}",
                   transform=a.transAxes, fontsize=6, color=col,
                   fontweight="bold", ha="left", va="top", zorder=6)
        t.set_path_effects([path_effects.withStroke(linewidth=2, foreground="white")])
        if r == 0:
            a.set_title("what it shows\n$12\\times20$ tiles", loc="left", fontsize=7)

    for i in (1, 3, 5):
        a = fig.add_subplot(gs[0, i])
        a.axis("off")
        a.annotate("", xy=(0.95, 0.55), xytext=(0.05, 0.55), xycoords="axes fraction",
                   arrowprops=dict(arrowstyle="-|>", color="#39414f", lw=1.1))

    # Two lines of about eighty characters. Wider than that and the text is
    # centered on a box narrower than itself, so both ends fall off the canvas.
    fig.supxlabel(
        rf"{len(obs)} {args.person}s disagree; their attention splits into modes ranked by support."
        "\n"
        r"Top-1 trains the primary region, the rest the auxiliary ones.",
        fontsize=6.5)

    print(f"[graphical-abstract] replay {replay} frame {args.frame}: "
          f"{len(hist)} stacked frames, {len(boxes)} regions, "
          f"scores {np.round(scores[:args.k], 3).tolist()}", flush=True)
    _save(fig, args.outdir, f"graphical_abstract_{replay}_{args.frame}{args.suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, parents=[common_parser()],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
