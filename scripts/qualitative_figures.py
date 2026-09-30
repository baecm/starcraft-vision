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
# (1) compare
# --------------------------------------------------------------------------

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

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
