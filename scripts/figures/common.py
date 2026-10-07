"""
figures/common.py
=================

What every figure script shares: the sys.path setup that makes src/ and
scripts/ importable, the style (sizes, colors, rcParams and the font), data
access (ground truth, predictions, raw frames, ranked modes), the drawing
helpers, the Director re-run behind the heatmap panels, and the argparse
options every figure takes (`common_parser`).

Moved out of scripts/qualitative_figures.py without changing behavior; the
only addition is the font selection below.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# scripts/figures/common.py -> scripts/ -> repository root. scripts/ is on the
# path for mode_disagreement (and for this package when a figure script is run
# directly), the root and src/ for the project modules, in the order the old
# single script put them.
figures_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(figures_dir)
root_dir = os.path.dirname(scripts_dir)
src_dir = os.path.join(root_dir, "src")
for _p in (scripts_dir, root_dir, src_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, patches
from scipy.ndimage import gaussian_filter

import config
from evaluate import load_coco_gt, load_coco_preds
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
# ESWA double-column width by default. FIG_WIDTH_MM sets the full-width canvas
# for another document: the thesis text block is 155 mm, so its figures are
# drawn at 155 mm and print at the font sizes below instead of at 82 % of them.
DOUBLE_COL = float(os.environ.get("FIG_WIDTH_MM", "190")) * MM
SINGLE_COL = 88 * MM    # one column of the same two-column layout


def legend_ncol(n: int) -> int:
    """Legend columns for n entries: one row on the 190 mm canvas, two below it,
    where a one-row legend of five or six entries runs past the figure edge."""
    return n if DOUBLE_COL >= 180 * MM else -(-n // 2)

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

# One Arial-like sans-serif for all text, mathtext included, so that labels
# typeset as $...$ do not switch to DejaVu in the middle of a line. Arial is
# not in the Docker image; Liberation Sans (fonts-liberation) is metrically
# compatible with it and is what the image is expected to provide.
FONT_CANDIDATES = ("Arial", "Liberation Sans", "Helvetica", "Nimbus Sans")


def _find_font_family(candidates=FONT_CANDIDATES) -> Optional[str]:
    """First family in `candidates` that matplotlib can resolve without falling
    back to its default font, or None."""
    for family in candidates:
        try:
            font_manager.findfont(font_manager.FontProperties(family=family),
                                  fallback_to_default=False)
        except Exception:
            continue
        return family
    return None


FONT_FAMILY = _find_font_family()
if FONT_FAMILY is not None:
    plt.rcParams.update({
        # DejaVu Sans stays behind it only for glyphs the family lacks.
        "font.sans-serif": [FONT_FAMILY, "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_FAMILY,
        "mathtext.it": f"{FONT_FAMILY}:italic",
        "mathtext.bf": f"{FONT_FAMILY}:bold",
        "mathtext.sf": FONT_FAMILY,
        # The custom fontset sends \mathcal to the "cursive" family, which the
        # image does not have, so Y_t, M_t, P_t came out as plain upright
        # letters. cmsy10 ships with matplotlib and holds TeX's calligraphic
        # capitals; the default fontset drew them from STIX.
        "mathtext.cal": "cmsy10",
    })
    print(f"[figures] font: {FONT_FAMILY} (text and mathtext)", flush=True)
else:
    print(f"[figures] WARNING: none of {', '.join(FONT_CANDIDATES)} is installed; "
          f"figures will fall back to DejaVu Sans. The Docker image needs the "
          f"fonts-liberation package (infra/docker/Dockerfile, then `make build`).",
          flush=True)


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


def _axes_rect(host, x, y, w, h):
    """Figure-fraction rect for a sub-axes placed in `host`'s data coordinates."""
    x0, x1 = host.get_xlim()
    y0, y1 = host.get_ylim()
    return [(x - x0) / (x1 - x0), (y - y0) / (y1 - y0),
            w / (x1 - x0), h / (y1 - y0)]


# --------------------------------------------------------------------------
# Director re-run (heatmap, graphical abstract)
# --------------------------------------------------------------------------

def _figure_device() -> "torch.device":
    """CUDA if this PyTorch build has kernels for the GPU, otherwise the CPU.

    `torch.cuda.is_available()` is true on a GPU newer than the build (an
    RTX 5090, sm_120, under the cu126 wheels), and the first kernel then fails
    with "no kernel image is available". A figure runs one frame, so the CPU
    costs nothing that matters.
    """
    import torch

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if f"sm_{major}{minor}" in torch.cuda.get_arch_list():
            return torch.device("cuda")
        print(f"[figures] GPU sm_{major}{minor} is not in this PyTorch build; using the CPU",
              flush=True)
    return torch.device("cpu")


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
    device = _figure_device()
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


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

SPEC_HELP = "NAME=MODEL_NAME[:EPOCH][@THRESHOLD], as in mode_disagreement.py"


def common_parser() -> argparse.ArgumentParser:
    """The options every figure and selection command takes, as a parent parser."""
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
    common.add_argument("--person", choices=["spectator", "observer"], default="observer",
                        help="what the figure calls a human viewer (thesis: spectator, paper: observer)")
    common.add_argument("--suffix", default="", help="appended to the output file stem")
    return common


# Names a figure prints for a model spec's NAME; anything else is printed with
# underscores as spaces.
DISPLAY_NAMES = {"maskrcnn": "Mask R-CNN", "director": "Director-CenterNet"}


def display_name(name: str) -> str:
    """The name a figure prints for a model spec's NAME."""
    return DISPLAY_NAMES.get(name, name.replace("_", " "))
