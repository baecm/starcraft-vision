#!/usr/bin/env python3
"""
qualitative_figures.py
======================

Draws the three qualitative figures of the Director-CenterNet paper
(Section "Qualitative Analysis") from real frames:

  compare     (1) one frame, three panels: the U observers and their ranked
                  modes / the baseline's top-K boxes / Director-CenterNet's
                  primary + auxiliary regions
  heatmap     (2) the same frame as four panels: coverage a_t / smoothed
                  coverage with ranked modes and support / predicted heatmap
                  Y_hat_t / decoded regions over the minimap
  trajectory  (3) the primary-region centre over a window of consecutive
                  frames for both methods, with the Top-1 / Top-2 mode tracks
                  and the tie frames (support margin 0) shaded

and a fourth subcommand that picks the frames:

  select      reads the frames_<name>.csv that mode_disagreement.py wrote for
              both methods and lists candidate frames for (1)/(2) and
              candidate windows for (3), together with the numbers the caption
              needs to say how the example was chosen.

A single frame cannot show that one method is better, and the paper's own
tables say Mask R-CNN leads on OC3@0.5 and that VD does not separate the
methods. `select` therefore ranks frames by how close they sit to a chosen
quantile of the per-frame difference, not by how flattering they are, and
prints where the chosen example falls so the caption can state it.

Predictions are read the way mode_disagreement.py reads them: COCO files under
{pred-root}/{model}/model_{epoch:03d}[_th<x>]/{replay}.rep/{label-method}.json,
with every box forced to the fixed viewport size anchored at its top-left.
The heatmap is not saved by inference, so `heatmap` re-runs the Director
checkpoint on that one frame and checks the decoded regions against the saved
predictions before drawing anything.

Usage
-----
Run inside the debugger container (it mounts scripts/, models/, predictions/,
results/ and data/):

  # 0. per-frame CSVs, if not already there
  make mode-disagreement-fg ARGS="--replays 275 1725 3613 4520 4664 \
      --model maskrcnn=<maskrcnn_run> --model director=dc_full_b16_f1_s456_v6 \
      --epoch 30 --outdir /workspace/results/mode_disagreement/fold1"

  # 1. candidates
  python3 /workspace/scripts/qualitative_figures.py select \
      --baseline-csv /workspace/results/mode_disagreement/fold1/frames_maskrcnn.csv \
      --director-csv /workspace/results/mode_disagreement/fold1/frames_director.csv

  # 2. figures
  python3 /workspace/scripts/qualitative_figures.py compare \
      --replay 1725 --frame 4312 \
      --baseline maskrcnn=<maskrcnn_run>:30 --director director=dc_full_b16_f1_s456_v6:30
  python3 /workspace/scripts/qualitative_figures.py heatmap \
      --replay 1725 --frame 4312 --director director=dc_full_b16_f1_s456_v6:30
  python3 /workspace/scripts/qualitative_figures.py trajectory \
      --replay 1725 --start 4200 --end 4500 \
      --baseline maskrcnn=<maskrcnn_run>:30 --director director=dc_full_b16_f1_s456_v6:30

or through make: make qualitative-figures-fg ARGS="compare --replay ...".

Figures go to --outdir (default /workspace/results/figures/qualitative) as
both .pdf and .png, named after the subcommand, replay and frame.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(root_dir, "src")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from scipy.ndimage import gaussian_filter

import config
from estimate import load_coco_gt, load_coco_preds
from metrics.modes import (
    analyse_method,
    box_center,
    box_mask,
    extract_modes,
    gt_boxes_by_frame,
    image_size,
    infer_region_size,
    predictions_from_dets,
)
from mode_disagreement import parse_model_spec

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

MM = 1 / 25.4
DOUBLE_COL = 190 * MM   # ESWA double-column width
SINGLE_COL = 88 * MM    # one column of the same two-column layout

C_MAIN = "#2458a6"
C_PIP = "#e07b24"
C_BASE = "#7a4fa8"
C_OBS = "#39414f"
C_TOP1 = C_MAIN
C_MINOR = C_PIP

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,    # embed TrueType, as journals ask
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})


def _save(fig, outdir: str, stem: str) -> None:
    """Write the figure at exactly DOUBLE_COL wide, in pdf and png.

    Not `bbox_inches="tight"`. That crops the canvas down to its content, so a
    figure declared at 190 mm came out at 148-175 mm depending on how much
    whitespace each layout happened to leave. The paper includes all three at
    `width=\\textwidth`, which then scales each by a different factor and lands
    the 8 pt tick labels anywhere between 8.7 and 10.3 pt on the page. Fitting
    the content into the fixed canvas instead keeps the scale at 1.0, so the
    font sizes set in rcParams are the sizes that print.
    """
    os.makedirs(outdir, exist_ok=True)
    if fig.get_layout_engine() is None:
        fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"[qualitative] wrote {path}", flush=True)
    plt.close(fig)


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------

def minimap_rgb(x: np.ndarray, fog: bool = True) -> np.ndarray:
    """Print-friendly RGB minimap from one raw (11, H, W) frame.

    The palette of notebooks/modules/visualization.create_reconstructed_minimap_rgb
    is built for a screen (dark ground, saturated units) and prints as a black
    square; this keeps its channel logic on a light ground. Masks are taken
    with `!= 0` per channel rather than `|` across channels, which the notebook
    version needs an integer dtype for.
    """
    ch = config.Channel
    h, w = x.shape[1:]
    rgb = np.empty((h, w, 3), dtype=float)
    rgb[:] = (0.933, 0.941, 0.910)
    rgb[x[ch.Terrain.value] > 0] = (0.851, 0.863, 0.812)
    rgb[x[ch.Resource.value] != 0] = (0.361, 0.612, 0.769)

    def any_of(*chs):
        return np.any(np.stack([x[c.value] != 0 for c in chs]), axis=0)

    p1 = any_of(ch.Player_1_Worker, ch.Player_1_Ground, ch.Player_1_Air, ch.Player_1_Building)
    p2 = any_of(ch.Player_2_Worker, ch.Player_2_Ground, ch.Player_2_Air, ch.Player_2_Building)
    rgb[p1] = (0.247, 0.561, 0.310)
    rgb[p2] = (0.753, 0.278, 0.247)
    if fog:
        rgb[x[ch.Vision.value] == 0] *= 0.82
    return rgb


def load_frame_npy(input_root: str, replay: str, frame: int) -> np.ndarray:
    path = os.path.join(input_root, f"{replay}.rep", f"{frame}.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"input frame not found: {path}")
    return np.load(path)


class Source:
    """One method's predictions for one replay, fixed-size boxes, score order."""

    def __init__(self, spec: str, args, replay: str, size_hw: Tuple[float, float]):
        self.name, self.model, self.epoch, self.threshold = parse_model_spec(spec, args.epoch)
        dets = load_coco_preds(args.pred_root, self.model, self.epoch, replay,
                               args.label_method, score_threshold=self.threshold)
        if not dets:
            raise FileNotFoundError(
                f"no predictions for {self.model} epoch {self.epoch} replay {replay}"
            )
        self.by_frame = predictions_from_dets(dets, (size_hw[1], size_hw[0]))
        suffix = "" if self.threshold is None else f"_th{self.threshold}"
        self.pred_dir = os.path.join(args.pred_root, self.model, f"model_{self.epoch:03d}{suffix}")

    def get(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.by_frame.get(frame, (np.empty((0, 4)), np.empty(0)))

    def provenance(self) -> dict:
        path = os.path.join(self.pred_dir, "inference_provenance.json")
        if not os.path.isfile(path):
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)


def load_gt(args, replay: str):
    coco = load_coco_gt(args.label_root, replay, args.label_method)
    gt = gt_boxes_by_frame(coco)
    height, width = image_size(coco)
    size_hw = infer_region_size(gt)
    return gt, height, width, size_hw


def frame_modes(obs: np.ndarray, height: int, width: int, args):
    return extract_modes(obs, height, width, sigma=args.sigma, min_sep=args.min_sep,
                         rel_threshold=args.rel_threshold, max_modes=args.max_modes)


def coverage_maps(obs: np.ndarray, height: int, width: int, sigma: float):
    """a_t and its smoothed field, computed exactly as extract_modes does
    (which keeps both local)."""
    cov = np.zeros((height, width), dtype=float)
    for b in obs:
        cov += box_mask(b, height, width)
    cov /= max(1, len(obs))
    return cov, gaussian_filter(cov, sigma=sigma, mode="constant")


# --------------------------------------------------------------------------
# Drawing helpers (tile coordinates: x = column, y = row, origin top-left)
# --------------------------------------------------------------------------

def _map_axes(ax, height: int, width: int) -> None:
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _rect(ax, box, color, lw=1.2, ls="-", alpha=1.0, z=3):
    x, y, w, h = box
    ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor=color,
                                   linewidth=lw, linestyle=ls, alpha=alpha, zorder=z))


def _label(ax, x, y, text, color, size=7):
    t = ax.text(x, y, text, color=color, fontsize=size, fontweight="bold",
                ha="center", va="center", zorder=6)
    t.set_path_effects([path_effects.withStroke(linewidth=2, foreground="white")])


def _draw_modes(ax, modes, with_support: bool = True) -> None:
    for i, (c, n) in enumerate(zip(modes.centers, modes.support)):
        color = C_TOP1 if i == 0 else C_MINOR
        ax.scatter([c[1]], [c[0]], s=22, color=color, edgecolor="white",
                   linewidth=0.8, zorder=5)
        if with_support:
            _label(ax, c[1], c[0] - 5, f"#{i + 1}  n={n}", color, size=6.5)


def _draw_regions(ax, boxes, k: int, primary_color: str, aux_color: str, number=True):
    for r, b in enumerate(boxes[:k]):
        color = primary_color if r == 0 else aux_color
        _rect(ax, b, color, lw=1.8 if r == 0 else 1.4)
        if number:
            _label(ax, b[0] + 2.5, b[1] + 2.5, str(r + 1), color, size=6.5)



# --------------------------------------------------------------------------
# select
# --------------------------------------------------------------------------

def cmd_select(args) -> None:
    oc = f"OC{args.k}@{args.delta}"
    keep = ["replay", "frame", "n_modes", "margin", "answered", "n_pred", oc, "top2_flip"]
    base = pd.read_csv(args.baseline_csv, usecols=lambda c: c in keep)
    dire = pd.read_csv(args.director_csv, usecols=lambda c: c in keep)
    df = base.merge(dire, on=["replay", "frame"], suffixes=("_base", "_dir"))
    df = df.rename(columns={"n_modes_base": "n_modes", "margin_base": "margin"})
    print(f"[select] {len(df)} frames in both CSVs", flush=True)

    # ---- (1)/(2): one frame with the requested number of modes, both answered
    pool = df[(df["n_modes"] == args.n_modes) & (df["answered_base"] == 1)
              & (df["answered_dir"] == 1) & (df["n_pred_dir"] >= 2)].copy()
    if pool.empty:
        print(f"[select] no frame with n_modes={args.n_modes} answered by both")
    else:
        pool["diff"] = pool[f"{oc}_dir"] - pool[f"{oc}_base"]
        target = pool["diff"].quantile(args.quantile)
        pool["dist"] = (pool["diff"] - target).abs()
        print()
        print(f"(1)/(2) pool: {len(pool)} frames with n_modes={args.n_modes}, both answered, "
              f"Director emitting >= 2 regions")
        print(f"  {oc} Director - baseline: mean {pool['diff'].mean():+.3f}, "
              f"median {pool['diff'].median():+.3f}; Director ahead on "
              f"{(pool['diff'] > 0).mean():.1%}, tied {(pool['diff'] == 0).mean():.1%}, "
              f"behind {(pool['diff'] < 0).mean():.1%}")
        print(f"  frames at the {args.quantile:.0%} quantile of the difference ({target:+.3f}):")
        cols = ["replay", "frame", "margin", f"{oc}_base", f"{oc}_dir", "n_pred_base", "n_pred_dir"]
        print(pool.sort_values(["dist", "replay", "frame"]).head(args.top)[cols].to_string(index=False))
        print("  Caption: say the frame was drawn at this quantile and give the three shares above.")

    # ---- (3): runs of consecutive tie frames
    runs = []
    for replay, g in df.sort_values("frame").groupby("replay"):
        step = g["frame"].diff()
        modal = step.mode().iloc[0] if len(step.mode()) else np.nan
        tie = (g["margin"] == 0).to_numpy()
        adjacent = (step == modal).to_numpy()
        frames = g["frame"].to_numpy()
        fb = g["top2_flip_base"].to_numpy()
        fd = g["top2_flip_dir"].to_numpy()
        start = None
        for i in range(len(g) + 1):
            inside = i < len(g) and tie[i] and (start is None or adjacent[i])
            if inside and start is None:
                start = i
            elif not inside and start is not None:
                n = i - start
                if n >= args.min_run:
                    runs.append({
                        "replay": replay, "start": int(frames[start]), "end": int(frames[i - 1]),
                        "frames": n,
                        "flips_base": int(np.nansum(fb[start:i])),
                        "flips_dir": int(np.nansum(fd[start:i])),
                    })
                start = i if (i < len(g) and tie[i]) else None
    print()
    if not runs:
        print(f"(3) no run of >= {args.min_run} consecutive margin-0 frames")
        return
    rdf = pd.DataFrame(runs)
    all_tie = df[df["margin"] == 0]
    print(f"(3) {len(rdf)} runs of >= {args.min_run} consecutive margin-0 frames. "
          f"Fold-wide top-2 flip rate on margin-0 frames: baseline "
          f"{all_tie['top2_flip_base'].mean():.4f}, Director {all_tie['top2_flip_dir'].mean():.4f}")
    rdf["rate_base"] = rdf["flips_base"] / rdf["frames"]
    print("  longest runs (pad the window with --pad frames either side when plotting):")
    print(rdf.sort_values("frames", ascending=False).head(args.top).to_string(index=False))
    print("  Caption: give this window's flip counts next to the fold-wide rates, so the "
          "example is read against the average and not as one.")


# --------------------------------------------------------------------------
# teaser
# --------------------------------------------------------------------------

def cmd_teaser(args) -> None:
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
        patches.Patch(fill=False, edgecolor=C_OBS, linestyle="--", label="observer viewport"),
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
    _save(fig, args.outdir, f"qual0_teaser_{replay}_{args.frame}")


# --------------------------------------------------------------------------
# (1) compare
# --------------------------------------------------------------------------

def cmd_export_panels(args) -> None:
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


def cmd_architecture(args) -> None:
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

    cy = 13.2                    # centre line the row sits on
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
         "cell + offset $\\rightarrow$ centre"])

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


def _axes_rect(host, x, y, w, h):
    """Figure-fraction rect for a sub-axes placed in `host`'s data coordinates."""
    x0, x1 = host.get_xlim()
    y0, y1 = host.get_ylim()
    return [(x - x0) / (x1 - x0), (y - y0) / (y1 - y0),
            w / (x1 - x0), h / (y1 - y0)]


def cmd_graphical_abstract(args) -> None:
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
    a1.set_title("Director-CenterNet\ncentre heatmap", loc="left", fontsize=7)

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
    # centred on a box narrower than itself, so both ends fall off the canvas.
    fig.supxlabel(
        rf"{len(obs)} observers disagree; their attention splits into modes ranked by support."
        "\n"
        r"Top-1 trains the primary region, the rest the auxiliary ones.",
        fontsize=6.5)

    print(f"[graphical-abstract] replay {replay} frame {args.frame}: "
          f"{len(hist)} stacked frames, {len(boxes)} regions, "
          f"scores {np.round(scores[:args.k], 3).tolist()}", flush=True)
    _save(fig, args.outdir, f"graphical_abstract_{replay}_{args.frame}")


def cmd_supervision_grid(args) -> None:
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
    titles = ["observer viewports", r"coverage $a_t$",
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
    _save(fig, args.outdir, f"qual6_supervision_grid_{nrow}")


def cmd_select_supervision(args) -> None:
    """Rank frames by how legibly they show the supervision construction.

    What makes a frame readable in the `supervision` figure is not how well any
    model does on it - no model appears - but the geometry of the ground truth:
    enough modes to show a ranking, a Top-1 that is not tied with the next one,
    and modes far enough apart that the panels do not look like one blob. Those
    come from the observer viewports alone, so this reads no predictions and
    needs no GPU.

    Frames are sampled at `--step`, because consecutive frames are near
    duplicates and scoring all of them buys nothing.
    """
    rows = []
    for replay in args.replays:
        replay = str(replay)
        gt, height, width, size_hw = load_gt(args, replay)
        frames = sorted(gt)[::args.step]
        for f in frames:
            obs = gt[f]
            m = frame_modes(obs, height, width, args)
            n = len(m.centers)
            if n < args.min_modes:
                continue
            sup = [int(s) for s in m.support]
            margin = sup[0] - sup[1] if n >= 2 else sup[0]
            if margin < args.min_margin:
                continue
            sep = min(float(np.linalg.norm(m.centers[i] - m.centers[j]))
                      for i in range(n) for j in range(i + 1, n))
            if sep < args.min_gap:
                continue
            rows.append({"replay": replay, "frame": int(f), "n_modes": n,
                         "support": "/".join(str(s) for s in sup),
                         "margin": margin, "min_sep": round(sep)})
        print(f"  {replay}: {len(frames)} frames sampled", flush=True)

    if not rows:
        print("\nno frame met the criteria; loosen --min-gap or --min-margin")
        return
    df = pd.DataFrame(rows).sort_values(["min_sep", "margin"], ascending=False)

    # One per replay first, so the shortlist is not five frames of one game.
    best = df.groupby("replay", as_index=False).head(args.per_replay)
    best = best.sort_values(["min_sep", "margin"], ascending=False).head(args.top)
    pd.set_option("display.width", 200)
    print(f"\n{len(df)} frames met the criteria; best {len(best)}, "
          f"at most {args.per_replay} per replay:\n")
    print(best.to_string(index=False))
    print("\nRender one with:  supervision --replay R --frame F")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"wrote the full list to {args.out}")


def cmd_supervision(args) -> None:
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
    ax[0].set_title(f"observer viewports, $U = {len(obs)}$", loc="left", fontsize=7)

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
    _save(fig, args.outdir, f"qual4_supervision_{replay}_{args.frame}")


def cmd_compare(args) -> None:
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
    titles = [f"(a) Human observers, U = {len(obs)}",
              f"(b) {base.name}, top-{args.k}",
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
        patches.Patch(fill=False, edgecolor=C_OBS, linestyle="--", label="observer viewport"),
        plt.Line2D([], [], marker="o", ls="", color=C_TOP1, label="Top-1 mode"),
        plt.Line2D([], [], marker="o", ls="", color=C_MINOR, label="minority mode"),
        patches.Patch(fill=False, edgecolor=C_BASE, label=f"{base.name} box"),
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
    _save(fig, args.outdir, f"qual1_compare_{replay}_{args.frame}")


# --------------------------------------------------------------------------
# (2) heatmap
# --------------------------------------------------------------------------

def _director_heatmap(args, dire: Source, replay: str, frame: int):
    """Re-run the Director checkpoint on one frame; returns the sigmoid map,
    the decoded boxes (xyxy) and scores, and the stride."""
    import torch

    from dataset.inference_dataset import InferenceDataset
    from inference import _load_model

    prov = dire.provenance()
    window = int(prov.get("window_size") or args.window_size)
    comps = prov.get("include_components") or args.include_components
    tau = float(prov.get("conf_threshold") or args.conf_threshold)
    k_max = int(prov.get("k_max") or args.k)

    ds = InferenceDataset(args.input_root, [replay], window_size=window, include_components=comps)
    idx = next((i for i, (r, f, _) in enumerate(ds.indexes) if f == frame), None)
    if idx is None:
        raise KeyError(f"frame {frame} is not the last frame of any window in replay {replay}")
    x, _ = ds[idx]

    ckpt = os.path.join(args.model_root, dire.model, f"model_{dire.epoch:03d}.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(ckpt, device, in_channels=x.shape[0], window_size=window,
                        architecture="director_centernet", k_max=k_max, conf_threshold=tau)
    with torch.no_grad():
        feat = model._extract_features(x.unsqueeze(0).to(device))
        hm, _logits, off, wh = model._predict_heads(feat)
        boxes, scores, *_ = model._extract_predicted_regions_differentiable(
            hm, off, wh, x.shape[1], x.shape[2], conf_thresh=tau, k_max=k_max)
    boxes, scores = boxes[0].cpu().numpy(), scores[0].cpu().numpy()
    # inference applies score_threshold after decoding, so the saved file can
    # hold fewer regions than the decoder returns; filter the same way
    th = prov.get("score_threshold")
    if th is not None:
        keep = scores >= float(th)
        boxes, scores = boxes[keep], scores[keep]
    return hm[0, 0].cpu().numpy(), boxes, scores, model.down_ratio, tau


def cmd_heatmap(args) -> None:
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
    cb.set_label(f"(b) normalised to its maximum; (c) sigmoid score, 1/U = {1 / len(obs):.1f}",
                 fontsize=6.5)
    axes[3].set_xlabel(rf"$\tau$ = {tau:g}, stride {stride}", fontsize=6.5)
    print(f"[heatmap] replay {replay} frame {args.frame}: modes support "
          f"{modes.support.tolist()}, peak scores {np.round(saved_scores, 3).tolist()}", flush=True)
    _save(fig, args.outdir, f"qual2_heatmap_{replay}_{args.frame}")


# --------------------------------------------------------------------------
# (3) trajectory
# --------------------------------------------------------------------------

def cmd_trajectory(args) -> None:
    replay = str(args.replay)
    gt, height, width, size_hw = load_gt(args, replay)
    frames = [f for f in sorted(gt) if args.start - args.pad <= f <= args.end + args.pad]
    if len(frames) < 4:
        raise ValueError(f"only {len(frames)} ground-truth frames in the window")
    base = Source(args.baseline, args, replay, size_hw)
    dire = Source(args.director, args, replay, size_hw)
    axis = 1 if args.axis == "x" else 0          # centres are (row, col)

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
            zorder=3, label=f"{base.name} top-1")
    if pip_f:
        ax.scatter(pip_f, pip_v, s=5, marker="s", color=C_PIP, lw=0, zorder=3,
                   label="Director-CenterNet auxiliary")
    ax.plot(fr, d_track, color=C_MAIN, lw=1.5, zorder=4, label="Director-CenterNet primary")
    ax.set_xlim(fr[0], fr[-1])
    ax.set_ylim(0, width if axis == 1 else height)
    ax.set_xlabel("frame")
    ax.set_ylabel(f"region centre {args.axis} (tiles)")
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.legend(loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.2))
    ax.text(1, 1.02, f"shaded: support margin 0 · top-2 flips {base.name} {flips['base']}, "
            f"Director {flips['dir']}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color="#5b6474")
    print(f"[trajectory] replay {replay} frames {frames[0]}-{frames[-1]} ({len(frames)}): "
          f"{sum(tie)} tie frames; top-2 flips {base.name} {flips['base']}, "
          f"Director {flips['dir']}", flush=True)
    _save(fig, args.outdir, f"qual3_trajectory_{replay}_{args.start}_{args.end}")


# --------------------------------------------------------------------------
# single-region failure modes (thesis, chapter on the single-region baseline)
# --------------------------------------------------------------------------

def cmd_select_single(args) -> None:
    """Candidates for `single-failure`, from one method's frames CSV.

    (a) frames with at least --n-modes attention modes that the model answered,
        ranked by distance from the pool's median OC1 and median IR - a typical
        multi-mode frame, not the worst one - and listed --per-replay at a time
        so that one replay cannot fill the list;
    (b) the longest runs of consecutive margin-0 frames, with the model's top-2
        flips inside each, next to the fold-wide flip rate on tied frames.
    """
    df = pd.read_csv(args.csv)
    oc = f"OC1@{args.delta}"
    print(f"[select-single] {len(df)} frames in {args.csv}", flush=True)

    pool = df[(df["n_modes"] >= args.n_modes) & (df["n_pred"] > 0)].copy()
    if pool.empty:
        print(f"(a) no answered frame with n_modes >= {args.n_modes}")
    else:
        # OC1 takes only the values k/U, so thousands of frames sit exactly on
        # its median; the IR term breaks that tie towards a typical frame.
        target = pool[oc].median()
        target_ir = pool["IR"].median()
        pool["dist"] = (pool[oc] - target).abs() + (pool["IR"] - target_ir).abs()
        print()
        print(f"(a) pool: {len(pool)} answered frames with n_modes >= {args.n_modes}; "
              f"{oc} mean {pool[oc].mean():.3f}, median {target:.3f}; "
              f"IR mean {pool['IR'].mean():.3f}, median {target_ir:.3f}")
        cols = ["replay", "frame", "progression", "n_modes", "support_top1", "support_top2",
                "IR", oc, "nearest_mode_rank"]
        best = (pool.sort_values(["dist", "replay", "frame"])
                .groupby("replay", sort=False).head(args.per_replay))
        print(best.head(args.top)[[c for c in cols if c in best]].to_string(index=False))

    runs = []
    for replay, g in df.sort_values("frame").groupby("replay"):
        step = g["frame"].diff()
        modal = step.mode().iloc[0] if len(step.mode()) else np.nan
        tie = (g["margin"] == 0).to_numpy()
        adjacent = (step == modal).to_numpy()
        frames = g["frame"].to_numpy()
        flip = g["top2_flip"].to_numpy()
        start = None
        for i in range(len(g) + 1):
            inside = i < len(g) and tie[i] and (start is None or adjacent[i])
            if inside and start is None:
                start = i
            elif not inside and start is not None:
                if i - start >= args.min_run:
                    runs.append({"replay": replay, "start": int(frames[start]),
                                 "end": int(frames[i - 1]), "frames": i - start,
                                 "flips": int(np.nansum(flip[start:i]))})
                start = i if (i < len(g) and tie[i]) else None
    print()
    if not runs:
        print(f"(b) no run of >= {args.min_run} consecutive margin-0 frames")
        return
    rdf = pd.DataFrame(runs)
    rdf["rate"] = rdf["flips"] / rdf["frames"]
    tied = df[df["margin"] == 0]
    print(f"(b) {len(rdf)} runs of >= {args.min_run} margin-0 frames; fold-wide top-2 flip "
          f"rate on margin-0 frames {tied['top2_flip'].mean():.4f}")
    print(rdf.sort_values("frames", ascending=False).head(args.top).to_string(index=False))
    print("  Caption: give the chosen run's flip rate next to the fold-wide one.")


def cmd_single_failure(args) -> None:
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
    ax.set_xlabel(f"region serves {served * u:.0f} of {u} spectators", fontsize=7)

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
        patches.Patch(fill=False, edgecolor=C_OBS, linestyle="--", label="spectator viewport"),
        plt.Line2D([], [], marker="o", ls="", color=C_TOP1, label="Top-1 mode"),
        plt.Line2D([], [], marker="o", ls="", color=C_MINOR, label="minority / Top-2 mode"),
        plt.Line2D([], [], color=C_BASE, lw=1.4, label=f"{src.name} region"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    print(f"[single-failure] (a) replay {replay} frame {args.frame}: {len(modes.centers)} "
          f"modes, support {modes.support.tolist()}, serves {served * u:.0f}/{u}; "
          f"(b) replay {t_replay} frames {frames[0]}-{frames[-1]} ({len(frames)}): "
          f"{sum(tie)} tied, {flips} top-2 flips", flush=True)
    _save(fig, args.outdir,
          f"single_failure_{replay}_{args.frame}_{t_replay}_{args.start}_{args.end}")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--label-root", default="/workspace/data/label/dst")
    common.add_argument("--label-method", default="all_correct")
    common.add_argument("--input-root", default="/workspace/data/input/dst")
    common.add_argument("--pred-root", default="/workspace/predictions")
    common.add_argument("--model-root", default="/workspace/models")
    common.add_argument("--outdir", default="/workspace/results/figures/qualitative")
    common.add_argument("--epoch", type=int, default=30,
                        help="default epoch for model specs that omit :EPOCH")
    common.add_argument("--k", type=int, default=3, help="regions drawn / scored per frame")
    common.add_argument("--delta", type=float, default=0.5)
    common.add_argument("--sigma", type=float, default=config.MODE_EXTRACTION_SIGMA)
    common.add_argument("--min-sep", type=float, default=config.MODE_EXTRACTION_MIN_SEP)
    common.add_argument("--rel-threshold", type=float, default=config.MODE_EXTRACTION_REL_THRESHOLD)
    common.add_argument("--max-modes", type=int, default=config.MODE_EXTRACTION_MAX_MODES)
    common.add_argument("--no-fog", action="store_true", help="do not darken unseen tiles")

    sub = ap.add_subparsers(dest="cmd", required=True)
    spec = "NAME=MODEL_NAME[:EPOCH][@THRESHOLD], as in mode_disagreement.py"

    p = sub.add_parser("select", parents=[common], help="list candidate frames and windows")
    p.add_argument("--baseline-csv", required=True, help="frames_<baseline>.csv from mode_disagreement.py")
    p.add_argument("--director-csv", required=True, help="frames_<director>.csv from mode_disagreement.py")
    p.add_argument("--n-modes", type=int, default=3)
    p.add_argument("--quantile", type=float, default=0.5,
                   help="quantile of the per-frame OC difference to draw from")
    p.add_argument("--min-run", type=int, default=6, help="shortest tie run listed for (3)")
    p.add_argument("--top", type=int, default=15)
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("teaser", parents=[common],
                       help="the opening figure: observers and ranked modes, one column wide")
    p.add_argument("--replay", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.set_defaults(func=cmd_teaser)

    p = sub.add_parser("export-panels", parents=[common],
                       help="the architecture figure's data panels, as separate images")
    p.add_argument("--replay", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--director", default=None, metavar="SPEC",
                   help=spec + "; omit to skip the panel with the regions drawn")
    p.add_argument("--stack-delta", type=int, default=8)
    p.add_argument("--stack", type=int, default=3)
    p.add_argument("--scale", type=int, default=8,
                   help="integer upscale of the raw maps, nearest-neighbour")
    p.set_defaults(func=cmd_export_panels)

    p = sub.add_parser("architecture", parents=[common],
                       help="the inference schematic")
    p.add_argument("--replay", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--director", required=True, metavar="SPEC", help=spec)
    p.add_argument("--stack-delta", type=int, default=8)
    p.add_argument("--stack", type=int, default=3)
    p.set_defaults(func=cmd_architecture)

    p = sub.add_parser("graphical-abstract", parents=[common],
                       help="the journal's graphical abstract, 13 x 5 cm")
    p.add_argument("--replay", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--director", required=True, metavar="SPEC", help=spec)
    p.add_argument("--stack-delta", type=int, default=8,
                   help="temporal stride of the stack (--delta is the overlap threshold)")
    p.add_argument("--stack", type=int, default=3, help="past frames in the stack")
    p.add_argument("--window-size", type=int, default=4)
    p.add_argument("--include-components", nargs="+",
                   default=["worker", "ground", "air", "building", "vision"])
    p.add_argument("--conf-threshold", type=float, default=0.1)
    p.set_defaults(func=cmd_graphical_abstract)

    p = sub.add_parser("supervision-grid", parents=[common],
                       help="several frames through the supervision construction")
    p.add_argument("--frames", nargs="+", required=True, metavar="REPLAY:FRAME")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--render-sigma", type=float, default=config.DIRECTOR_RENDER_SIGMA)
    p.set_defaults(func=cmd_supervision_grid)

    p = sub.add_parser("select-supervision", parents=[common],
                       help="rank frames by how legibly they show the supervision")
    p.add_argument("--replays", nargs="+", required=True)
    p.add_argument("--min-modes", type=int, default=3)
    p.add_argument("--min-margin", type=int, default=1,
                   help="support of the Top-1 mode minus the Top-2's; 1 excludes ties")
    # Not --min-sep: that is the extraction parameter D, the floor the mode
    # finder enforces. This is the separation the frame actually achieved.
    p.add_argument("--min-gap", type=float, default=30.0,
                   help="smallest pairwise distance between the extracted modes, in tiles")
    p.add_argument("--step", type=int, default=20, help="frame sampling stride")
    p.add_argument("--per-replay", type=int, default=3)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--out", default=None, help="write the full ranked list here")
    p.set_defaults(func=cmd_select_supervision)

    p = sub.add_parser("supervision", parents=[common],
                       help="the supervision row of the architecture figure")
    p.add_argument("--replay", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--stride", type=int, default=4,
                   help="output stride the target is rendered at")
    p.add_argument("--render-sigma", type=float, default=config.DIRECTOR_RENDER_SIGMA,
                   help="Gaussian sigma of the rendered target, in tiles")
    p.set_defaults(func=cmd_supervision)

    p = sub.add_parser("compare", parents=[common], help="figure (1)")
    p.add_argument("--replay", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--baseline", required=True, metavar="SPEC", help=spec)
    p.add_argument("--director", required=True, metavar="SPEC", help=spec)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("heatmap", parents=[common], help="figure (2)")
    p.add_argument("--replay", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--director", required=True, metavar="SPEC", help=spec)
    p.add_argument("--window-size", type=int, default=4,
                   help="used only if inference_provenance.json does not record it")
    p.add_argument("--include-components", nargs="+",
                   default=["worker", "ground", "air", "building", "vision"],
                   help="used only if inference_provenance.json does not record it")
    p.add_argument("--conf-threshold", type=float, default=0.1,
                   help="tau; used only if inference_provenance.json does not record it")
    p.set_defaults(func=cmd_heatmap)

    p = sub.add_parser("trajectory", parents=[common], help="figure (3)")
    p.add_argument("--replay", required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--pad", type=int, default=0, help="frame ids added either side")
    p.add_argument("--axis", choices=["x", "y"], default="x")
    p.add_argument("--baseline", required=True, metavar="SPEC", help=spec)
    p.add_argument("--director", required=True, metavar="SPEC", help=spec)
    p.set_defaults(func=cmd_trajectory)

    p = sub.add_parser("select-single", parents=[common],
                       help="candidates for single-failure, from one frames CSV")
    p.add_argument("--csv", required=True, help="frames_<method>.csv from mode_disagreement.py")
    p.add_argument("--n-modes", type=int, default=4, help="fewest modes for (a)")
    p.add_argument("--per-replay", type=int, default=3, help="frames listed per replay for (a)")
    p.add_argument("--min-run", type=int, default=6, help="shortest tie run listed for (b)")
    p.add_argument("--top", type=int, default=15)
    p.set_defaults(func=cmd_select_single)

    p = sub.add_parser("single-failure", parents=[common],
                       help="the two failure modes of a single-region observer")
    p.add_argument("--replay", required=True, help="replay of the multi-mode frame (a)")
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--tie-replay", required=True, help="replay of the tie run (b)")
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--pad", type=int, default=0, help="frame ids added either side")
    p.add_argument("--axis", choices=["x", "y"], default="x")
    p.add_argument("--baseline", required=True, metavar="SPEC", help=spec)
    p.set_defaults(func=cmd_single_failure)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
