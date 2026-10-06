"""
figures/architecture.py
=======================

The inference schematic drawn in matplotlib: stacked input, backbone, heads,
ranked multi-peak extraction and the decoded region set, with real data at
both ends. Writes qual5_architecture_<replay>_<frame>.{pdf,png}.

  make figure-fg FIG=architecture ARGS="--replay 4520 --frame 10298 \
      --director director=dc_full_b16_f1_s456_v6:30"

Same as `qualitative_figures.py architecture`.
"""

from __future__ import annotations

import argparse
import os
import sys

_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from figures.common import (
    C_MAIN,
    C_PIP,
    DOUBLE_COL,
    MM,
    SPEC_HELP,
    Source,
    _axes_rect,
    _draw_regions,
    _map_axes,
    _save,
    common_parser,
    load_frame_npy,
    load_gt,
    minimap_rgb,
    patches,
    plt,
)

HELP = "the inference schematic"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--replay", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--director", required=True, metavar="SPEC", help=SPEC_HELP)
    p.add_argument("--stack-delta", type=int, default=8)
    p.add_argument("--stack", type=int, default=3)


def run(args) -> None:
    """The inference schematic: tensor, backbone, heads, extraction, regions.

    Drawn in one axes with explicit coordinates rather than a gridspec. A block
    diagram is a set of boxes at chosen positions, and a layout engine that
    moves them is working against the drawing; the only thing it buys is
    automatic sizing, which a fixed canvas does not need.

    The two ends carry real data - the stacked frames the network reads and the
    region set it returns - so the figure's claims about what goes in and comes
    out are checkable. Everything between them is a schematic of code and has
    nothing to show.
    """
    replay = str(args.replay)
    gt, height, width, size_hw = load_gt(args, replay)
    dire = Source(args.director, args, replay, size_hw)
    boxes, _ = dire.get(args.frame)
    bg = minimap_rgb(load_frame_npy(args.input_root, replay, args.frame), fog=not args.no_fog)
    hist = []
    for j in range(args.stack, -1, -1):
        try:
            hist.append(minimap_rgb(
                load_frame_npy(args.input_root, replay, args.frame - j * args.stack_delta),
                fog=not args.no_fog))
        except FileNotFoundError:
            pass

    INK, EDGE, FILL = "#2b3038", "#9aa3ad", "#eef2f7"
    fig = plt.figure(figsize=(DOUBLE_COL, 46 * MM))
    # Every axes here is placed by hand, so declare a layout engine to stop
    # _save falling back to tight_layout and moving them.
    fig.set_layout_engine("none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 24)
    ax.axis("off")

    def box(x, y, w, h, title, lines, fc=FILL, ec=EDGE, tc=INK):
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6",
            linewidth=0.7, edgecolor=ec, facecolor=fc, zorder=2))
        ax.text(x + 0.7, y + h - 0.9, title, fontsize=6.6, fontweight="bold",
                color=tc, va="top", ha="left", zorder=3)
        if lines:
            ax.text(x + 0.7, y + h - 2.6, "\n".join(lines), fontsize=5.4,
                    color="#4a525c", va="top", ha="left", linespacing=1.45, zorder=3)

    def arrow(x0, x1, y, label=None):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0), zorder=4)
        if label:
            ax.text((x0 + x1) / 2, y + 0.5, label, fontsize=5.4, color="#4a525c",
                    ha="center", va="bottom", zorder=4)

    def imbox(x, y, s, img, label=None, regions=False):
        a = fig.add_axes(_axes_rect(ax, x, y, s, s))
        a.imshow(img, extent=(0, width, height, 0), interpolation="nearest")
        if regions:
            _draw_regions(a, boxes, args.k, C_MAIN, C_PIP)
        _map_axes(a, height, width)
        for sp in a.spines.values():
            sp.set_color(EDGE); sp.set_linewidth(0.7)
        if label:
            ax.text(x + s / 2, y - 0.9, label, fontsize=5.6, color="#4a525c",
                    ha="center", va="top", zorder=4)
        return a

    cy = 13.2                    # center line the row sits on
    s = 13.5                     # side of the square image panels
    stack_drop = (len(hist) - 1) * 0.9 if hist else 0.0

    # (1) the stacked input
    for i, im in enumerate(hist):
        d = (len(hist) - 1 - i) * 0.9
        a = fig.add_axes(_axes_rect(ax, 1.5 + d, cy - s / 2 - d, s, s))
        a.imshow(im, extent=(0, width, height, 0), interpolation="nearest")
        _map_axes(a, height, width)
        for sp in a.spines.values():
            sp.set_color(EDGE); sp.set_linewidth(0.6)
        a.set_zorder(i)
    # Below the *lowest* frame of the stack, not below the front one: the
    # offset puts the oldest frame further down than the newest.
    ax.text(1.5 + s / 2, cy - s / 2 - stack_drop - 0.9,
            f"$X^{{\\mathrm{{stk}}}}_t$, $(L{{+}}1)C \\times 128 \\times 128$\n"
            f"$\\Delta = {args.stack_delta}$, $L = {args.stack}$",
            fontsize=5.6, color="#4a525c", ha="center", va="top", zorder=4)

    arrow(17.5, 21.5, cy)

    # (2) backbone
    box(21.5, cy - 6.2, 15, 12.4, "ResNet-50 + FPN",
        ["all stages trained", "read at the finest level, $P_2$"])
    # Bars live in the lower third of the box: the title and its two lines take
    # the top, and a bar taller than that runs out through the bottom edge.
    for i, (bw, bh) in enumerate([(1.1, 4.6), (1.4, 3.8), (1.7, 3.0), (2.0, 2.2)]):
        ax.add_patch(patches.Rectangle((23.2 + i * 2.7, cy - 2.2 - bh / 2),
                                       bw, bh, facecolor="#c8d6e8",
                                       edgecolor="#8fa6c4", lw=0.4, zorder=3))

    arrow(36.5, 41.0, cy, r"$s = 4$")

    # (3) heads
    for i, (nm, sub) in enumerate([("Heatmap head", r"$\hat{\mathcal{Y}}_t$, $32\times32$"),
                                   ("Offset head", r"$2 \times 32 \times 32$"),
                                   ("Size head", r"$2 \times 32 \times 32$")]):
        yy = cy + 4.4 - i * 4.6
        box(41.0, yy - 1.9, 15.5, 3.8, nm, [sub],
            ec=C_MAIN if i == 0 else EDGE)

    arrow(56.5, 61.0, cy)

    # (4) extraction
    box(61.0, cy - 6.6, 20, 13.2, "Ranked multi-peak extraction",
        [r"$3\times3$ max-pool suppression",
         "drop the border ring",
         r"keep score $> \tau$,  $\tau < 1/U$",
         r"top-$K$, descending",
         "cell + offset $\\rightarrow$ center"])

    arrow(81.0, 85.0, cy)

    # (5) the region set
    imbox(85.0, cy - s / 2, s, bg, regions=True)
    ax.text(85.0 + s / 2, cy - s / 2 - 1.2,
            "ordered set $\\hat{\\mathcal{P}}_t$\n"
            "1: primary   2, 3: auxiliary",
            fontsize=5.6, color="#4a525c", ha="center", va="top", zorder=4)

    ax.text(0.8, 23.2, "INFERENCE  ·  SINGLE FORWARD PASS", fontsize=6.2,
            fontweight="bold", color="#6b7682", ha="left", va="top")
    ax.plot([0.8, 99.2], [22.2, 22.2], color="#d6dce3", lw=0.6, zorder=0)

    print(f"[architecture] replay {replay} frame {args.frame}: "
          f"{len(hist)} stacked frames, {len(boxes[:args.k])} regions drawn", flush=True)
    _save(fig, args.outdir, f"qual5_architecture_{replay}_{args.frame}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, parents=[common_parser()],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
