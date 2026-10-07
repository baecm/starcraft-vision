"""
Shared pieces of the analysis scripts in this directory.

Every analysis script answers its question from the same three inputs: the
observers' viewports (ground truth), a model's stored predictions, and the
ranked attention modes extracted from the viewports. This module holds the
code that reads those inputs and the few computations that several scripts
must do identically:

  ModelSpec / parse_model_spec   NAME=MODEL[:EPOCH][@THRESHOLD] on the CLI
  load_replay_gt                 one replay's viewports, map size, viewport size
  load_regions                   one model's regions per frame, highest score first
  per_viewport_set               compute something once per distinct set of
                                 viewports (frames repeat their viewports often)
  classify_regions               new / dup / off against a frame's ranked modes
  weighted_slope, frame_slope    the count response beta_n

Conventions used throughout:
  - A box is [x, y, w, h] in tiles with (x, y) its top-left corner.
  - A mode center is (row, col) in tiles.
  - A region is a predicted box forced to the observers' viewport size and
    anchored at its stored top-left corner, which is what the rest of the
    project means by a predicted viewport (metrics.modes.predictions_from_dets).

Importing this module puts src/ and the repository root on sys.path, so a
script only needs `from analysis_common import ...` after its stdlib imports.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config  # noqa: E402
from evaluate import load_coco_gt, load_coco_preds  # noqa: E402
from metrics.modes import (  # noqa: E402
    _rect_overlap_ratio,
    gt_boxes_by_frame,
    image_size,
    infer_region_size,
    predictions_from_dets,
)

# The test replays of fold 1, the fold every thesis analysis is reported on.
FOLD1_TEST_REPLAYS = ["275", "1725", "3613", "4520", "4664"]

# beta_n is fitted over frames with 1..MAX_MODES_FOR_SLOPE modes, the most
# the extraction keeps (config.MODE_EXTRACTION_MAX_MODES).
MAX_MODES_FOR_SLOPE = 5


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

class ModelSpec(NamedTuple):
    """Which predictions to read: <pred-root>/<model>/model_<epoch>[_th<threshold>]/.

    `name` is only the label the script prints. `threshold` stays the string
    the user typed, because it is spliced into the directory name as is
    (`@0.5` reads `model_030_th0.5`, `@0.50` would read `model_030_th0.50`).
    """
    name: str
    model: str
    epoch: int
    threshold: Optional[str]


def parse_model_spec(spec: str, default_epoch: int) -> ModelSpec:
    """NAME=MODEL[:EPOCH][@THRESHOLD] -> ModelSpec.

    The threshold names the prediction directory to read, `model_NNN_th<x>`,
    which is where inference puts a run that set one. A sweep is therefore
    addressed here rather than by copying directories: comparing a proposal
    detector against a heatmap one is only meaningful at a matched region
    count, and for the detector that count is set by this filter.
    """
    name, sep, rest = spec.partition("=")
    if not sep:
        raise ValueError(
            f"--model spec must be NAME=MODEL_NAME[:EPOCH][@THRESHOLD], got: {spec!r}"
        )
    rest, _, threshold = rest.partition("@")
    model, _, epoch = rest.partition(":")
    return ModelSpec(name, model, int(epoch) if epoch else default_epoch, threshold or None)


def parse_run_spec(spec: str) -> Tuple[str, Optional[str]]:
    """MODEL[@THRESHOLD] -> (model, threshold), for options naming one run."""
    model, _, threshold = spec.partition("@")
    return model, (threshold or None)


def add_replay_args(ap: argparse.ArgumentParser, *, required: bool = False) -> None:
    """--replays, --label-root, --label-method: where the ground truth is."""
    if required:
        ap.add_argument("--replays", type=str, nargs="+", required=True,
                        help="replay ids, pooled into one result")
    else:
        ap.add_argument("--replays", nargs="+", default=list(FOLD1_TEST_REPLAYS),
                        help="replay ids, pooled into one result (default: fold 1 test)")
    ap.add_argument("--label-root", default="/workspace/data/label/dst")
    ap.add_argument("--label-method", default="all_correct")


def add_prediction_args(ap: argparse.ArgumentParser) -> None:
    """--pred-root, --epoch: where the predictions are."""
    ap.add_argument("--pred-root", default="/workspace/predictions")
    ap.add_argument("--epoch", type=int, default=30,
                    help="checkpoint epoch, for model specs that omit :EPOCH")


def add_mode_args(ap: argparse.ArgumentParser) -> None:
    """The mode extraction parameters of metrics.modes.extract_modes.

    Defaults are the ones every reported number uses (src/config.py).
    """
    ap.add_argument("--sigma", type=float, default=config.MODE_EXTRACTION_SIGMA,
                    help="Gaussian sigma (tiles) for smoothing the coverage map")
    ap.add_argument("--min-sep", type=float, default=config.MODE_EXTRACTION_MIN_SEP,
                    help="minimum separation D between modes (tiles)")
    ap.add_argument("--rel-threshold", type=float, default=config.MODE_EXTRACTION_REL_THRESHOLD,
                    help="peak floor as a fraction of the frame's maximum")
    ap.add_argument("--max-modes", type=int, default=config.MODE_EXTRACTION_MAX_MODES)


def mode_params(args: argparse.Namespace) -> Tuple[float, float, float, int]:
    """(sigma, min_sep, rel_threshold, max_modes) in extract_modes' order."""
    return args.sigma, args.min_sep, args.rel_threshold, args.max_modes


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class ReplayGT:
    """One replay's ground truth."""
    replay: str
    coco: object
    height: int                                # map size in tiles
    width: int
    viewports_by_frame: Dict[int, np.ndarray]  # frame -> (U, 4) observer boxes

    @cached_property
    def viewport_hw(self) -> Tuple[float, float]:
        """The observers' viewport size (h, w) in tiles; every mode region and
        every predicted region is this size."""
        return infer_region_size(self.viewports_by_frame)

    @property
    def viewport_wh(self) -> Tuple[float, float]:
        h, w = self.viewport_hw
        return w, h


def load_replay_gt(label_root: str, replay: str, label_method: str) -> ReplayGT:
    coco = load_coco_gt(label_root, replay, label_method)
    height, width = image_size(coco)
    return ReplayGT(str(replay), coco, height, width, gt_boxes_by_frame(coco))


def load_regions(
    pred_root: str,
    model: str,
    epoch: int,
    replay: str,
    label_method: str,
    threshold: Optional[str],
    size_wh: Optional[Tuple[float, float]],
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """frame -> (boxes, scores) of one model, highest score first.

    `size_wh` forces every box to the viewport size at its top-left corner;
    None keeps the stored size (only for reproducing older numbers).
    """
    dets = load_coco_preds(pred_root, model, epoch, replay, label_method, threshold)
    return predictions_from_dets(dets, size_wh)


def per_viewport_set(
    viewports_by_frame: Dict[int, np.ndarray],
    compute: Callable[[np.ndarray], object],
) -> Dict[int, object]:
    """frame -> compute(viewports), calling `compute` once per distinct set
    of viewports. Observers often hold still, so consecutive frames repeat
    the same boxes and mode extraction is the expensive step."""
    cache: Dict[bytes, object] = {}
    out: Dict[int, object] = {}
    for frame, viewports in viewports_by_frame.items():
        key = viewports.tobytes()
        if key not in cache:
            cache[key] = compute(viewports)
        out[frame] = cache[key]
    return out


# ---------------------------------------------------------------------------
# Shared computations
# ---------------------------------------------------------------------------

def classify_regions(
    boxes: Sequence[np.ndarray],
    mode_boxes: Sequence[np.ndarray],
    height: int,
    width: int,
    delta: float,
) -> List[str]:
    """Label each region, in the order given (highest score first):

      new  it covers (>= delta of its area) a mode no earlier region covers
      dup  the mode it covers best is already covered by an earlier region
      off  it covers no mode at delta

    so the number of distinct modes served is the count of "new". A region
    marks every mode it covers at delta as covered, not only its best one.
    `mode_boxes` must not be empty.
    """
    covered: set = set()
    kinds: List[str] = []
    for box in boxes:
        coverage = np.array([_rect_overlap_ratio(box, mb, height, width) for mb in mode_boxes])
        best = int(np.argmax(coverage))
        if coverage[best] < delta:
            kinds.append("off")
        elif best in covered:
            kinds.append("dup")
        else:
            kinds.append("new")
        covered |= set(np.nonzero(coverage >= delta)[0].tolist())
    return kinds


def weighted_slope(points: Sequence[Tuple[float, float, float]]) -> float:
    """Least-squares slope of y on n over (n, mean_y_at_n, frames_at_n),
    each point weighted by its frame count. On per-frame data this equals the
    plain slope over frames (frame_slope); it is the form of beta_n in the
    thesis (Eq. countslope)."""
    total = sum(f for _, _, f in points)
    mean_n = sum(n * f for n, _, f in points) / total
    mean_y = sum(y * f for _, y, f in points) / total
    num = sum(f * (n - mean_n) * (y - mean_y) for n, y, f in points)
    den = sum(f * (n - mean_n) ** 2 for n, _, f in points)
    return num / den if den else float("nan")


def frame_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y on x over frames."""
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())
