"""
figures/export_panels.py
========================

The architecture figure's data panels as separate, axis-free images (the
stacked input frames, the plain map and, with --director, the map with the
decoded regions), for placing in the hand-drawn schematic.

  make figure-fg FIG=export_panels ARGS="--replay 4520 --frame 10298 \
      --director director=dc_full_b16_f1_s456_v6:30"

Same as `qualitative_figures.py export-panels`.
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
    SPEC_HELP,
    Source,
    _draw_regions,
    _map_axes,
    common_parser,
    load_frame_npy,
    load_gt,
    minimap_rgb,
    plt,
)

HELP = "the architecture figure's data panels, as separate images"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--replay", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--director", default=None, metavar="SPEC",
                   help=SPEC_HELP + "; omit to skip the panel with the regions drawn")
    p.add_argument("--stack-delta", type=int, default=8)
    p.add_argument("--stack", type=int, default=3)
    p.add_argument("--scale", type=int, default=8,
                   help="integer upscale of the raw maps, nearest-neighbour")


def run(args) -> None:
    """Write the architecture figure's data panels as separate images.

    For drawing the schematic by hand: the boxes and arrows are better placed
    in a vector editor, but the frames the network reads and the regions it
    returns should still be the real ones rather than redrawn. This writes them
    with no axes, frame, title or padding, so each file is the map and nothing
    else.

    The raw maps are upscaled by an integer factor with nearest-neighbour
    replication. A 128 x 128 image placed in an editor and stretched gets
    smoothed, which turns the tile grid into mush; replicating first means the
    editor has nothing left to interpolate.
    """
    replay = str(args.replay)
    gt, height, width, size_hw = load_gt(args, replay)
    os.makedirs(args.outdir, exist_ok=True)
    k = args.scale

    written = []

    def save_raw(name, rgb):
        big = np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)
        path = os.path.join(args.outdir, name)
        plt.imsave(path, np.clip(big, 0, 1))
        written.append((name, f"{big.shape[1]}x{big.shape[0]}"))

    # The stack the network reads, oldest first, named in that order.
    for i, j in enumerate(range(args.stack, -1, -1)):
        f = args.frame - j * args.stack_delta
        try:
            raw = load_frame_npy(args.input_root, replay, f)
        except FileNotFoundError:
            print(f"  [!] frame {f} missing, skipped", flush=True)
            continue
        save_raw(f"arch_input_{i}_t-{j * args.stack_delta}.png",
                 minimap_rgb(raw, fog=not args.no_fog))

    bg = minimap_rgb(load_frame_npy(args.input_root, replay, args.frame),
                     fog=not args.no_fog)
    save_raw("arch_map_plain.png", bg)

    if args.director:
        dire = Source(args.director, args, replay, size_hw)
        boxes, scores = dire.get(args.frame)

        # Drawn through matplotlib because it has rectangles; the axes fills the
        # figure exactly, so the file is still the map and nothing else.
        px = width * k
        fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
        fig.set_layout_engine("none")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(bg, extent=(0, width, height, 0), interpolation="nearest")
        _draw_regions(ax, boxes, args.k, C_MAIN, C_PIP)
        _map_axes(ax, height, width)
        ax.set_frame_on(False)
        path = os.path.join(args.outdir, "arch_map_regions.png")
        fig.savefig(path, dpi=100, transparent=False)
        plt.close(fig)
        written.append(("arch_map_regions.png", f"{px}x{px}"))
        print(f"[export-panels] regions: {len(boxes[:args.k])} drawn, "
              f"scores {np.round(scores[:args.k], 3).tolist()}", flush=True)

    print(f"\n[export-panels] replay {replay} frame {args.frame} -> {args.outdir}")
    for n, s in written:
        print(f"  {n:38s} {s}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, parents=[common_parser()],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
