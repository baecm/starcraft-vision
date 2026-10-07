"""
Human attention-mode diagnostics.
==================================

Given COCO-style viewport files for one replay (one ground-truth file with U
observer viewports per frame, plus one or more prediction files), this module
answers *where* a single-region observer's disagreement with the aggregated
human ground truth comes from:

  (1) Error attribution by attention mode (`attribute`)
      For every frame, the primary prediction is attributed to one of:
        served_top1   - it covers the dominant (Top-1) mode
        served_minor  - it covers a lower-ranked (minority) mode instead
        straddle      - it partially overlaps two modes and covers neither
        off_mode      - it is near no mode at all
      `straddle` is the direct signature of mode collapse; `off_mode` is
      ordinary model error. The split between them says whether a miss is
      caused by the multimodality of the target or by the model.

  (2) Margin-conditional disagreement (`by_margin`, `by_n_modes`)
      Miss rate binned by the number of extracted modes |M_t| and by the
      support margin n(m1) - n(m2). The hypothesis is that errors concentrate
      in low-margin frames.

  (3) Mode flip rate (`analyse_method` -> `mode_flip` / `top2_flip` columns)
      How often the *identity* of the ranked GT mode nearest the primary
      prediction changes between consecutive frames, split into flips within
      the top-2 modes (oscillation between competing regions) and other
      flips. This is deliberately *not* the same quantity as
      `evaluator.compute_m_cti`'s `jump_rate`:
        - `jump_rate` (compute_m_cti) is magnitude/threshold-based: it only
          asks whether the primary prediction's centre moved more than
          `jump_threshold` pixels between frames, with no reference to the
          human ground truth at all.
        - `mode_flip_rate` (here) is identity-based: it asks whether the
          nearest-mode *rank* changed, using the ranked modes extracted from
          the observer union in this frame. A prediction can travel a long
          distance while staying inside one broad mode (no flip, but a large
          jump_rate contribution), or cross a mode boundary with a small
          move near the boundary (a flip, with a negligible jump_rate
          contribution). The two are complementary, not redundant, so both
          are reported side by side (see `primary_track_m_cti` below) rather
          than one replacing the other.

  (4) Reference multi-region metrics (`analyse_method` -> `IR`, `I@delta`,
      `BoK{k}@delta`, `OC{k}@delta` columns)
      So the single-region result and the best-of-K result can be compared
      directly. `IR` is the same ratio as `custom_evaluator.intersection_ratio`
      (the function behind the project's `ic@000/030/050` numbers) - see
      `coverage_of`'s docstring for why it's computed by array slicing here
      instead of calling that function directly: this ratio runs tens of
      thousands of times per replay, and pycocotools' RLE encode/merge/area
      overhead dominated the whole analysis's runtime for masks this small.

Loading follows the rest of `src/metrics` and `src/evaluate.py`: ground truth
goes through `pycocotools.coco.COCO`, and predictions are grouped by
`image_id` the same way `evaluate.coco_to_kernel_labels` already does.
`scripts/mode_disagreement.py` resolves replay/model names to actual file
paths via `evaluate.load_coco_gt` / `evaluate.load_coco_preds` (the same
`{label-root}/{replay}.rep/{label-method}.json` and
`{pred-root}/{model}/model_{epoch}/{replay}.rep/{label-method}.json` layout
used by `make estimate`); `predictions_from_dets` adapts the latter's output
into the (boxes, scores) shape this module works with. Mode extraction
itself (Gaussian-smoothed peak finding over the observer union, ranked by
support) is new to this module - nothing else in `src/metrics` builds ranked,
multimodal attention regions from observer viewports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from scipy.ndimage import gaussian_filter, maximum_filter

from .evaluator import compute_m_cti

ATTRIBUTIONS = ["served_top1", "served_minor", "straddle", "off_mode", "no_mode"]


# --------------------------------------------------------------------------
# Loading (reuses pycocotools.coco.COCO + the project's usual preds_by_img
# grouping instead of a bespoke JSON parser)
# --------------------------------------------------------------------------

def image_size(coco: COCO) -> Tuple[int, int]:
    """(height, width) of a COCO dataset whose images must share one size."""
    sizes = {(im["height"], im["width"]) for im in coco.dataset.get("images", [])}
    if len(sizes) != 1:
        raise ValueError(f"mixed image sizes {sizes}")
    return sizes.pop()


def load_gt(path: str) -> Tuple[COCO, int, int]:
    """Load a ground-truth COCO file and return (coco, height, width)."""
    coco = COCO(path)
    height, width = image_size(coco)
    return coco, height, width


def gt_boxes_by_frame(coco: COCO) -> Dict[int, np.ndarray]:
    """frame -> (U, 4) array of observer viewport boxes [x, y, w, h]."""
    by_frame: Dict[int, np.ndarray] = {}
    for img in coco.dataset.get("images", []):
        image_id = int(img["id"])
        anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        if anns:
            by_frame[image_id] = np.array([a["bbox"] for a in anns], dtype=float)
    return by_frame


def _boxes_and_scores(
    dets: List[dict],
    size_wh: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    items = sorted(dets, key=lambda a: -float(a.get("score", 1.0)))
    boxes = np.array([a["bbox"] for a in items], dtype=float)
    scores = np.array([float(a.get("score", 1.0)) for a in items], dtype=float)
    if size_wh is not None and len(boxes):
        boxes[:, 2] = float(size_wh[0])
        boxes[:, 3] = float(size_wh[1])
    return boxes, scores


def predictions_from_dets(
    dets_by_img: Dict[int, List[dict]],
    size_wh: Optional[Tuple[float, float]] = None,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Adapt the {image_id: [det, ...]} grouping returned by
    `evaluate.load_coco_preds` into the (boxes, scores) shape `analyse_method`
    expects, highest score first.

    `size_wh` anchors every predicted box at its stored top-left corner and
    forces it to that (width, height). A viewport is a fixed-size camera
    rectangle, so this is what the rest of the project means by a prediction -
    the detector's own width/height are not used downstream. Leaving it None
    keeps whatever the file stores, which skews any position derived from the
    box: a box clipped at the map edge by inference's clamp() comes back
    narrower, and `box_center` then reports a centre pulled toward that edge by
    (w_true - w_stored) / 2. Edge predictions are exactly the ones affected, so
    raw sizes systematically overstate how close predictions sit to a corner.
    """
    return {
        image_id: _boxes_and_scores(dets, size_wh)
        for image_id, dets in dets_by_img.items()
        if dets
    }


def load_predictions(
    path: str,
    size_wh: Optional[Tuple[float, float]] = None,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """frame -> (boxes, scores), loaded directly from a prediction json file.

    Mirrors the preds_by_img grouping already used by
    `evaluate.coco_to_kernel_labels`. Prefer
    `evaluate.load_coco_preds` + `predictions_from_dets` when the file lives
    at the project's usual `predictions/<model>/model_<epoch>/<replay>.rep/`
    layout, since that also handles score-threshold suffixed folders.
    See `predictions_from_dets` for `size_wh`.
    """
    with open(path) as fh:
        data = json.load(fh)
    anns = data["annotations"] if isinstance(data, dict) else data

    by_img: Dict[int, List[dict]] = {}
    for a in anns:
        by_img.setdefault(int(a["image_id"]), []).append(a)
    return predictions_from_dets(by_img, size_wh)


def infer_region_size(gt_by_frame: Dict[int, np.ndarray], sample: int = 200) -> Tuple[float, float]:
    """Median observer viewport size (h, w), used as the canonical mode size."""
    obs = np.concatenate(list(gt_by_frame.values())[:sample])
    return float(np.median(obs[:, 3])), float(np.median(obs[:, 2]))


# --------------------------------------------------------------------------
# Geometry helpers (all in tile units; bbox is [x, y, w, h])
# --------------------------------------------------------------------------

def box_slices(box: np.ndarray, height: int, width: int):
    x, y, w, h = box
    x0 = int(round(max(0.0, x)))
    y0 = int(round(max(0.0, y)))
    x1 = int(round(min(float(width), x + w)))
    y1 = int(round(min(float(height), y + h)))
    return slice(y0, max(y0, y1)), slice(x0, max(x0, x1))


def box_center(box: np.ndarray) -> np.ndarray:
    x, y, w, h = box
    return np.array([y + h / 2.0, x + w / 2.0])  # (row, col)


def box_from_center(center: np.ndarray, size_hw, height: int, width: int) -> np.ndarray:
    """A canonical region of the fixed viewport size centred on `center`,
    shifted (not cropped) to stay inside the map."""
    h, w = size_hw
    y = float(np.clip(center[0] - h / 2.0, 0, max(0, height - h)))
    x = float(np.clip(center[1] - w / 2.0, 0, max(0, width - w)))
    return np.array([x, y, w, h], dtype=float)


def box_mask(box: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    ys, xs = box_slices(box, height, width)
    mask[ys, xs] = True
    return mask


def _slice_coverage(ys: slice, xs: slice, mask: np.ndarray) -> float:
    """|box_slice ∩ mask| / |box_slice| for an already-computed box slice."""
    area = (ys.stop - ys.start) * (xs.stop - xs.start)
    if area == 0:
        return 0.0
    return float(mask[ys, xs].sum()) / float(area)


def coverage_of(box: np.ndarray, mask: np.ndarray, height: int, width: int) -> float:
    """|box ∩ mask| / |box| - the same ratio as
    `custom_evaluator.intersection_ratio(box, mask, denom="pred")` (the
    function behind `ic@000/030/050`), computed by slicing the existing
    boolean array instead of going through pycocotools RLE encode/merge/area.
    This runs O(frames * modes * k) times in `analyse_method`'s per-frame
    loop - tens of thousands of frames per replay - where RLE's C-call and
    compression overhead (on masks this small and already dense) was the
    dominant cost of the whole analysis. The ratio itself is identical
    either way; only the implementation changed, for speed.
    """
    ys, xs = box_slices(box, height, width)
    return _slice_coverage(ys, xs, mask)


def _rect_overlap_ratio(box_a: np.ndarray, box_b: np.ndarray, height: int, width: int) -> float:
    """|box_a ∩ box_b| / |box_a| for two axis-aligned boxes, from clipped
    slice bounds directly - no (H, W) array ever gets allocated, since both
    sides are plain rectangles (unlike `coverage_of`, whose second argument
    is a genuine, non-rectangular mask such as an observer union)."""
    ya, xa = box_slices(box_a, height, width)
    yb, xb = box_slices(box_b, height, width)
    area_a = (ya.stop - ya.start) * (xa.stop - xa.start)
    if area_a == 0:
        return 0.0
    inter_y = max(0, min(ya.stop, yb.stop) - max(ya.start, yb.start))
    inter_x = max(0, min(xa.stop, xb.stop) - max(xa.start, xb.start))
    return float(inter_y * inter_x) / float(area_a)


# --------------------------------------------------------------------------
# Ranked attention modes
# --------------------------------------------------------------------------

@dataclass
class Modes:
    centers: np.ndarray      # (m, 2) mode centres as (row, col)
    support: np.ndarray      # (m,) number of observers covering each centre
    union: np.ndarray        # (H, W) bool, hard union of observer viewports
    observers: list          # list of (H, W) bool masks, one per observer
    n_observers: int


def extract_modes(
    obs_boxes: np.ndarray,
    height: int,
    width: int,
    sigma: float,
    min_sep: float,
    rel_threshold: float,
    max_modes: int,
) -> Modes:
    n_obs = len(obs_boxes)
    obs_masks = [box_mask(b, height, width) for b in obs_boxes]

    coverage = np.zeros((height, width), dtype=float)
    for mask in obs_masks:
        coverage += mask
    coverage /= max(1, n_obs)
    union = coverage > 0

    smoothed = gaussian_filter(coverage, sigma=sigma, mode="constant")
    if smoothed.max() <= 0:
        return Modes(np.zeros((0, 2)), np.zeros(0, dtype=int), union, obs_masks, n_obs)

    # local maxima of the smoothed field, above a relative floor
    peak = smoothed >= maximum_filter(smoothed, size=3, mode="constant")
    peak &= smoothed >= rel_threshold * smoothed.max()
    rows, cols = np.nonzero(peak)
    if len(rows) == 0:
        return Modes(np.zeros((0, 2)), np.zeros(0, dtype=int), union, obs_masks, n_obs)

    cands = np.stack([rows, cols], axis=1).astype(float)
    # order by smoothed value, then greedily suppress anything within min_sep
    order = np.argsort(-smoothed[rows, cols])
    kept: list[np.ndarray] = []
    for idx in order:
        c = cands[idx]
        if all(np.linalg.norm(c - k) >= min_sep for k in kept):
            kept.append(c)
    centers = np.array(kept)

    # support = number of observers whose viewport covers the mode centre
    support = np.array(
        [
            sum(int(mask[int(round(c[0])), int(round(c[1]))]) for mask in obs_masks)
            for c in centers
        ],
        dtype=int,
    )

    # rank by support, breaking ties by smoothed response
    resp = np.array([smoothed[int(round(c[0])), int(round(c[1]))] for c in centers])
    rank = np.lexsort((-resp, -support))
    centers, support = centers[rank][:max_modes], support[rank][:max_modes]

    return Modes(centers, support, union, obs_masks, n_obs)


# --------------------------------------------------------------------------
# Per-frame analysis
# --------------------------------------------------------------------------

def attribute(
    primary: np.ndarray,
    modes: Modes,
    size_hw,
    height: int,
    width: int,
    delta: float,
    straddle_floor: float,
) -> tuple[str, int, float]:
    """Attribute the primary prediction to an attention mode.

    Returns (attribution, nearest_mode_rank, best_mode_coverage).
    `nearest_mode_rank` is 0-based over the support-ranked modes, -1 if none.
    """
    if len(modes.centers) == 0:
        return "no_mode", -1, 0.0

    covs = np.array(
        [
            _rect_overlap_ratio(primary, box_from_center(c, size_hw, height, width), height, width)
            for c in modes.centers
        ]
    )
    dists = np.linalg.norm(modes.centers - box_center(primary), axis=1)
    nearest = int(np.argmin(dists))

    best = int(np.argmax(covs))
    if covs[best] >= delta:
        return ("served_top1" if best == 0 else "served_minor"), nearest, float(covs[best])

    # covers no single mode: does it sit between two of them?
    partial = np.sort(covs)[::-1]
    if len(covs) >= 2 and partial[1] >= straddle_floor:
        return "straddle", nearest, float(covs[best])

    return "off_mode", nearest, float(covs[best])


def _region_metrics(
    boxes: np.ndarray,
    modes: Modes,
    height: int,
    width: int,
    delta: float,
    k_max: int,
    suffix: str = "",
) -> dict:
    """IR, I@delta and the per-k BoK/OC figures for one set of predicted boxes.

    Factored out so the same arithmetic scores a frame's own prediction and,
    on a frame the model declined, the prediction carried forward from the last
    frame it did answer. `suffix` is what keeps the two families apart.
    """
    primary = boxes[0]
    ir = coverage_of(primary, modes.union, height, width)
    out = {
        f"IR{suffix}": ir,
        f"I@{delta}{suffix}": float(ir >= delta),
    }

    # box_slices(b) is the same for a given b across every k that includes it
    # (top_k only grows), so compute each box's slice once per frame instead
    # of once per (k, observer) - this and the switch away from RLE are what
    # make the loop tractable on 60k+-frame replays.
    top_slices = [box_slices(b, height, width) for b in boxes[:k_max]]

    for k in range(1, k_max + 1):
        slices_k = top_slices[:k]
        out[f"BoK{k}@{delta}{suffix}"] = float(
            any(_slice_coverage(ys, xs, modes.union) >= delta for ys, xs in slices_k)
        )
        served = 0
        for mask in modes.observers:
            if not mask.any():
                continue
            obs_area = float(mask.sum())
            # fraction of this observer's own viewport covered by box b,
            # i.e. intersection_ratio(b, mask, denom="gt")
            best = max(
                (float(mask[ys, xs].sum()) / obs_area for ys, xs in slices_k),
                default=0.0,
            )
            served += int(best >= delta)
        out[f"OC{k}@{delta}{suffix}"] = served / max(1, modes.n_observers)
    return out


def _unanswered_row(
    frame: int,
    first_frame: int,
    span: int,
    modes: Modes,
    delta: float,
    k_max: int,
    drop_observer: Optional[int] = None,
) -> dict:
    """A row for a frame on which the model emitted no region.

    The ground truth is known either way, so the mode count and the support
    figures are filled in. Everything that needs a prediction is left missing,
    and `answered` is 0. `n_pred` is 0 rather than missing, because emitting
    nothing is a count and the count-calibration figures should see it as one.
    `nearest_mode_rank` is -1 so that the mode-flip test, which requires a
    known rank on both sides of a step, skips any pair touching this frame.

    The `_ff` family starts missing too. The caller fills it from the last
    answered frame when there is one to carry.
    """
    row = {
        "frame": frame,
        "progression": (frame - first_frame) / max(1, span - 1),
        "n_modes": len(modes.centers),
        "support_top1": int(modes.support[0]) if len(modes.support) else 0,
        "support_top2": int(modes.support[1]) if len(modes.support) > 1 else 0,
        "n_pred": 0,
        "primary_score": np.nan,
        "IR": np.nan,
        f"I@{delta}": np.nan,
        "IR_ff": np.nan,
        f"I@{delta}_ff": np.nan,
        "attribution": None,
        "nearest_mode_rank": -1,
        "best_mode_coverage": np.nan,
        "primary_cy": np.nan,
        "primary_cx": np.nan,
        "primary_cy_ff": np.nan,
        "primary_cx_ff": np.nan,
        "answered": 0.0,
        "carried": np.nan,
    }
    row["margin"] = row["support_top1"] - row["support_top2"]
    for k in range(1, k_max + 1):
        row[f"BoK{k}@{delta}"] = np.nan
        row[f"OC{k}@{delta}"] = np.nan
        row[f"BoK{k}@{delta}_ff"] = np.nan
        row[f"OC{k}@{delta}_ff"] = np.nan
    if drop_observer is not None:
        row["dropped"] = drop_observer
    return row


def analyse_method(
    gt_by_frame: Dict[int, np.ndarray],
    pred_by_frame: Dict[int, Tuple[np.ndarray, np.ndarray]],
    height: int,
    width: int,
    *,
    size_hw,
    sigma: float,
    min_sep: float,
    rel_threshold: float,
    max_modes: int,
    delta: float,
    straddle_floor: float,
    k_max: int,
    label: str = "",
    drop_observer: Optional[int] = None,
) -> pd.DataFrame:
    # Every ground-truth frame is analysed, not only those the model answered.
    # The intersection of the two key sets used to be taken here, which removed
    # the frames a confidence threshold had emptied before any metric could see
    # them, and with them the denominator that makes coverage measurable.
    frames = sorted(gt_by_frame)
    if not frames:
        raise ValueError("ground truth has no frames")
    if not set(frames) & set(pred_by_frame):
        raise ValueError("ground truth and prediction share no frame ids")

    span = max(frames) - min(frames) + 1
    records = []

    total_frames = len(frames)
    # same cadence as evaluator.eval_intersection_run / MultiRegionEvaluator:
    # at most ~10 updates per run, never more often than every 5000 frames,
    # so a fold-wide log file doesn't balloon on large replays.
    step_interval = max(5000, total_frames // 10)
    tag = f" {label}" if label else ""

    # the most recent answered prediction, carried into frames the model
    # declines so that the _ff family has something to score
    last_boxes = None

    for idx, frame in enumerate(frames):
        if (idx + 1) % step_interval == 0 or (idx + 1) == total_frames:
            print(f"  [ModeDisagreement]{tag} {idx + 1}/{total_frames} frames "
                  f"({(idx + 1) / total_frames * 100:.1f}%)", flush=True)

        obs = gt_by_frame[frame]
        if drop_observer is not None:
            if len(obs) <= drop_observer:
                continue
            # One observer is removed from the ground truth, so every figure
            # this pass produces is measured against the rest. Published
            # numbers from a study that scored against fewer observers cannot
            # be read beside ours without this, since a smaller union makes
            # every overlap ratio smaller for a reason that has nothing to do
            # with the model.
            obs = np.delete(obs, drop_observer, axis=0)
        modes = extract_modes(
            obs, height, width,
            sigma=sigma, min_sep=min_sep,
            rel_threshold=rel_threshold, max_modes=max_modes,
        )

        boxes, scores = pred_by_frame.get(
            frame, (np.empty((0, 4), dtype=float), np.empty(0, dtype=float))
        )

        if len(boxes) == 0:
            # Nothing cleared the model's confidence threshold here. Keep the
            # frame instead of dropping it: it is the denominator `summarise`
            # needs for coverage, and declining to answer is exactly how a
            # raised threshold flatters itself - the frames it drops are the
            # crowded ones, so every average computed without them is higher
            # for a reason that has nothing to do with prediction quality.
            row = _unanswered_row(
                frame, frames[0], span, modes, delta, k_max,
                drop_observer=drop_observer,
            )
            if last_boxes is not None:
                row.update(_region_metrics(
                    last_boxes, modes, height, width, delta, k_max, suffix="_ff"
                ))
                held = box_center(last_boxes[0])
                row["primary_cy_ff"] = held[0]
                row["primary_cx_ff"] = held[1]
                row["carried"] = 1.0
            records.append(row)
            continue

        primary = boxes[0]
        attribution, nearest, mode_cov = attribute(
            primary, modes, size_hw, height, width, delta, straddle_floor
        )
        center = box_center(primary)

        row = {
            "frame": frame,
            "progression": (frame - frames[0]) / max(1, span - 1),
            "n_modes": len(modes.centers),
            "support_top1": int(modes.support[0]) if len(modes.support) else 0,
            "support_top2": int(modes.support[1]) if len(modes.support) > 1 else 0,
            "n_pred": len(boxes),
            "primary_score": float(scores[0]) if len(scores) else np.nan,
            "attribution": attribution,
            "nearest_mode_rank": nearest,
            "best_mode_coverage": mode_cov,
            "primary_cy": center[0],
            "primary_cx": center[1],
            "primary_cy_ff": center[0],
            "primary_cx_ff": center[1],
            "answered": 1.0,
            "carried": 0.0,
        }
        row["margin"] = row["support_top1"] - row["support_top2"]

        plain = _region_metrics(boxes, modes, height, width, delta, k_max)
        row.update(plain)
        # On a frame the model answered, the two families coincide. Copying
        # rather than recomputing keeps the _ff column defined on every row
        # without paying for the arithmetic twice.
        row.update({f"{key}_ff": value for key, value in plain.items()})

        if drop_observer is not None:
            row["dropped"] = drop_observer

        last_boxes = boxes
        records.append(row)

    df = pd.DataFrame(records).sort_values("frame").reset_index(drop=True)

    # temporal quantities, only across genuinely consecutive frames (a gap in
    # frame ids must not silently count as one "step" - see module docstring
    # on why this stays a local diff rather than a call into
    # evaluator.compute_m_cti, which has no such gap awareness)
    step = df["frame"].diff()
    modal_step = step.mode().iloc[0] if len(step.mode()) else np.nan
    adjacent = step == modal_step
    df["VD_step"] = np.hypot(
        df["primary_cy"].diff(), df["primary_cx"].diff()
    ).where(adjacent)
    # Under forward fill a declined frame holds the previous viewport, so its
    # displacement is zero. That makes VD_ff optimistic about stability in the
    # same measure that scoring a decline as 0 is pessimistic about accuracy;
    # the two are meant to be read as a pair, not chosen between.
    df["VD_ff_step"] = np.hypot(
        df["primary_cy_ff"].diff(), df["primary_cx_ff"].diff()
    ).where(adjacent)

    prev_rank = df["nearest_mode_rank"].shift()
    same_step = df["VD_step"].notna()
    valid = same_step & (prev_rank >= 0) & (df["nearest_mode_rank"] >= 0)
    df["mode_flip"] = np.where(valid, prev_rank != df["nearest_mode_rank"], np.nan)
    df["top2_flip"] = np.where(
        valid & prev_rank.isin([0, 1]) & df["nearest_mode_rank"].isin([0, 1]),
        prev_rank != df["nearest_mode_rank"],
        np.nan,
    )
    return df


def primary_track_m_cti(
    pred_by_frame: Dict[int, Tuple[np.ndarray, np.ndarray]],
    frames: List[int],
    jump_threshold: float = 35.0,
    carry_forward: bool = False,
    min_run: int = 4,
) -> Dict[str, float]:
    """Whole-sequence velocity/jerk/jump_rate of the primary (top-1)
    prediction, via `evaluator.compute_m_cti` - the project's existing
    magnitude/threshold-based instability metric. Reported next to
    `mode_flip_rate`/`top2_flip_rate` for comparison, not as a replacement:
    see the module docstring for why the two are not interchangeable.

    A frame the model did not answer breaks the trajectory rather than closing
    over it. Dropping such frames used to leave the two frames on either side
    adjacent in the array handed to `compute_m_cti`, which then read the jump
    across the gap as ordinary camera motion, so the reported jerk partly
    measured how often the model declined. The sequence is instead cut into
    runs of genuinely consecutive answered frames, each run scored on its own,
    and the runs combined in proportion to their length. Runs shorter than
    `min_run` are dropped, since jerk is a third difference and needs four
    points; `tracked_fraction` says how much of the sequence survived, and a
    low value means the figures describe only the stretches where the model
    kept answering.

    With `carry_forward`, an unanswered frame holds the previous viewport
    instead, which is what a live system does with a real camera. That keeps
    one unbroken run and scores a held camera as motionless, which flatters
    stability in the same measure that scoring a decline as a miss is harsh on
    accuracy. The two are meant to be read as a pair.
    """
    if len(frames) < 2:
        return {"m_cti": 0.0, "jerk": 0.0, "jump_rate": 0.0, "velocity": 0.0,
                "tracked_fraction": 0.0}

    step_counts: Dict[int, int] = {}
    for a, b in zip(frames, frames[1:]):
        step_counts[b - a] = step_counts.get(b - a, 0) + 1
    modal_step = max(step_counts, key=step_counts.get)

    runs: List[List[List[float]]] = []
    current: List[List[float]] = []
    prev_frame = None
    last_box = None

    for f in frames:
        boxes, _ = pred_by_frame.get(f, (np.empty((0, 4)), np.empty(0)))
        if len(boxes) == 0:
            if not (carry_forward and last_box is not None):
                runs.append(current)
                current = []
                prev_frame = None
                continue
            box = last_box
        else:
            box = boxes[0]
            last_box = box

        if prev_frame is not None and f - prev_frame != modal_step:
            runs.append(current)
            current = []

        x, y, w, h = box
        current.append([x, y, x + w, y + h])
        prev_frame = f

    runs.append(current)
    runs = [r for r in runs if len(r) >= min_run]

    keys = ("m_cti", "jerk", "jump_rate", "velocity")
    if not runs:
        empty = {k: 0.0 for k in keys}
        empty["tracked_fraction"] = 0.0
        return empty

    totals = {k: 0.0 for k in keys}
    weight = 0.0
    for run in runs:
        trajectory = np.array(run, dtype=float)[:, None, :]  # (T, 1, 4)
        res = compute_m_cti(trajectory, jump_threshold=jump_threshold,
                            box_format="xyxy")
        run_weight = float(len(run))
        weight += run_weight
        for k in keys:
            totals[k] += float(res[k]) * run_weight

    out = {k: totals[k] / weight for k in keys}
    out["tracked_fraction"] = weight / float(len(frames))
    return out

def _baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Rows from the pass that kept every observer in the ground truth.

    A --drop-one-observer run appends one pass per observer, each with that
    observer removed, so a pooled frame holds both families. Everything except
    the _drop1 keys is defined over the baseline pass alone, which is what
    keeps the flag from moving a number that was reported without it.
    """
    if "dropped" not in df.columns:
        return df
    return df[df["dropped"].isna()]


def _reduced(df: pd.DataFrame) -> pd.DataFrame:
    """Rows from the passes that removed one observer."""
    if "dropped" not in df.columns:
        return df.iloc[0:0]
    return df[df["dropped"].notna()]

def _answered(df: pd.DataFrame) -> pd.DataFrame:
    """Rows the model actually answered.

    `analyse_method` keeps a row for every ground-truth frame, including the
    ones where nothing cleared the confidence threshold, so that `summarise`
    can report coverage. Every breakdown below is defined over answered frames
    only, which is what it meant before unanswered rows were kept.
    """
    if "answered" not in df.columns:
        return df
    return df[df["answered"] == 1.0]

def summarise(df: pd.DataFrame, delta: float, k_max: int) -> dict:
    """Fold-level aggregate.

    Two families of numbers come back. The plain keys (`IR`, `I@d`, `BoK*`,
    `OC*`) average over the frames the model answered, which is how a model is
    usually described. The `_cov` keys score an unanswered frame as 0 and
    divide by every ground-truth frame instead.

    Report the second family, or restrict every method to a common frame set,
    whenever methods with different `coverage` are compared. Raising a
    confidence threshold deletes frames rather than improving predictions, and
    the frames it deletes are the crowded ones, so the plain keys reward a
    method for declining to answer where the task is hardest.

    Every key except the _drop1 family comes from the baseline pass, the one that
    kept all observers in the ground truth, so a --drop-one-observer run reports
    exactly what a plain run does and adds to it.
    """
    base = _baseline(df)
    ans = _answered(base)
    n_total = len(base)
    n_ans = len(ans)
    coverage = n_ans / n_total if n_total else 0.0

    out = {
        "frames": n_ans,
        "frames_total": n_total,
        "coverage": coverage,
        "IR": ans["IR"].mean(),
        f"I@{delta}": ans[f"I@{delta}"].mean(),
        "VD": ans["VD_step"].mean(),
        "mode_flip_rate": ans["mode_flip"].mean(),
        "top2_flip_rate": ans["top2_flip"].mean(),
        "mean_n_modes": ans["n_modes"].mean(),
    }
    out["IR_cov"] = out["IR"] * coverage
    out[f"I@{delta}_cov"] = out[f"I@{delta}"] * coverage

    for k in range(1, k_max + 1):
        out[f"BoK{k}@{delta}"] = ans[f"BoK{k}@{delta}"].mean()
        out[f"OC{k}@{delta}"] = ans[f"OC{k}@{delta}"].mean()
        out[f"BoK{k}@{delta}_cov"] = out[f"BoK{k}@{delta}"] * coverage
        out[f"OC{k}@{delta}_cov"] = out[f"OC{k}@{delta}"] * coverage

    # Count calibration. MAE_n and Exact_n are the level of the region count;
    # the responsiveness slope beta_n lives in scripts/budget_allocation.py,
    # since it needs the E[n_pred | n_modes] table rather than per-frame rows.
    # The level is the part a confidence threshold can buy outright, so the
    # _cov forms matter here for the same reason they do above. An unanswered
    # frame emits no region, so it enters them as n_pred = 0 rather than as a
    # missing value.
    err_ans = (ans["n_pred"] - ans["n_modes"]).abs()
    out["MAE_n"] = err_ans.mean()
    out["Exact_n"] = float((ans["n_pred"] == ans["n_modes"]).mean()) if n_ans else np.nan
    err_all = (base["n_pred"].fillna(0) - base["n_modes"]).abs()
    out["MAE_n_cov"] = err_all.mean()
    out["Exact_n_cov"] = float((base["n_pred"].fillna(0) == base["n_modes"]).mean()) if n_total else np.nan
    out["mean_n_pred"] = ans["n_pred"].mean()
    # The diagnostic that makes the coverage trap visible at a glance: when
    # mean_n_modes_all exceeds mean_n_modes, the frames the model declined
    # were the crowded ones, and every plain key above is flattered by their
    # absence.
    out["mean_n_modes_all"] = base["n_modes"].mean()

    # Reduced-observer protocol, from the passes that removed one. A study that
    # scores against fewer observers is not measuring the same thing: a smaller
    # union makes every overlap ratio smaller whatever the model does, so these
    # keys are what makes such a number comparable. On this corpus the gap is
    # about 0.05 of IR, larger than the differences between the methods being
    # compared, so it cannot be left as a footnote.
    red = _answered(_reduced(df))
    if len(red):
        out["drop1_frames"] = len(red)
        out["IR_drop1"] = red["IR"].mean()
        out[f"I@{delta}_drop1"] = red[f"I@{delta}"].mean()
        out["mean_n_modes_drop1"] = red["n_modes"].mean()
        for k in range(1, k_max + 1):
            out[f"BoK{k}@{delta}_drop1"] = red[f"BoK{k}@{delta}"].mean()
            out[f"OC{k}@{delta}_drop1"] = red[f"OC{k}@{delta}"].mean()

    # Forward fill: a declined frame holds the last viewport the model did
    # produce, which is what a live system does with a real camera. It is
    # scored against the ground truth of the frame it is held into, so a stale
    # camera is charged for being stale rather than deleted from the average.
    # Read it beside the _cov family: forward fill is the charitable treatment
    # of a decline and zero fill the harsh one, and the honest figure is
    # bracketed by the two. One caveat - VD_ff counts a held camera as
    # motionless, so it favours stability; the accuracy keys carry no such
    # favour.
    ff_scored = base["IR_ff"].notna()
    out["coverage_ff"] = float(ff_scored.mean()) if n_total else 0.0
    out["carried_frac"] = float((base["carried"] == 1.0).mean()) if n_total else 0.0
    out["IR_ff"] = base["IR_ff"].mean()
    out[f"I@{delta}_ff"] = base[f"I@{delta}_ff"].mean()
    out["VD_ff"] = base["VD_ff_step"].mean()
    for k in range(1, k_max + 1):
        out[f"BoK{k}@{delta}_ff"] = base[f"BoK{k}@{delta}_ff"].mean()
        out[f"OC{k}@{delta}_ff"] = base[f"OC{k}@{delta}_ff"].mean()

    for label in ATTRIBUTIONS:
        out[f"frac_{label}"] = float((ans["attribution"] == label).mean()) if n_ans else np.nan
    misses = ans[ans[f"I@{delta}"] == 0]
    out["miss_rate"] = float(len(misses)) / max(1, n_ans)
    for label in ATTRIBUTIONS:
        out[f"miss_{label}"] = (
            float((misses["attribution"] == label).mean()) if len(misses) else np.nan
        )
    return out


def by_quartile(df: pd.DataFrame, delta: float, k_max: int) -> pd.DataFrame:
    df = _baseline(df)
    q = pd.cut(df["progression"], [-0.001, 0.25, 0.5, 0.75, 1.0],
               labels=["1/4", "2/4", "3/4", "4/4"])
    rows = []
    for label, part in df.groupby(q, observed=True):
        row = {"progression": label}
        row.update(summarise(part, delta, k_max))
        rows.append(row)
    return pd.DataFrame(rows)


def by_margin(df: pd.DataFrame, delta: float) -> pd.DataFrame:
    rows = []
    for margin, part in _answered(_baseline(df)).groupby("margin", observed=True):
        rows.append({
            "margin": margin,
            "n_frames": len(part),
            "IR": part["IR"].mean(),
            f"I@{delta}": part[f"I@{delta}"].mean(),
            "frac_straddle": float((part["attribution"] == "straddle").mean()),
            "frac_served_minor": float((part["attribution"] == "served_minor").mean()),
            "top2_flip_rate": part["top2_flip"].mean(),
        })
    return pd.DataFrame(rows).sort_values("margin")


def by_replay(df: pd.DataFrame, delta: float) -> pd.DataFrame:
    """Per-replay breakdown of a fold-level `df` (one `analyse_method` call
    per replay, concatenated with a "replay" column) - lets a fold-wide
    mode_flip_rate/IR be checked for replays that dominate or skew it."""
    rows = []
    for replay, part in _baseline(df).groupby("replay", observed=True):
        ans = _answered(part)
        rows.append({
            "replay": replay,
            "n_frames": len(ans),
            "coverage": len(ans) / max(1, len(part)),
            "IR": ans["IR"].mean(),
            f"I@{delta}": ans[f"I@{delta}"].mean(),
            "mode_flip_rate": ans["mode_flip"].mean(),
            "top2_flip_rate": ans["top2_flip"].mean(),
            "frac_straddle": float((ans["attribution"] == "straddle").mean()),
            "frac_served_minor": float((ans["attribution"] == "served_minor").mean()),
        })
    return pd.DataFrame(rows).sort_values("replay")


def by_n_modes(df: pd.DataFrame, delta: float) -> pd.DataFrame:
    rows = []
    for n, part in _baseline(df).groupby("n_modes", observed=True):
        ans = _answered(part)
        rows.append({
            "n_modes": n,
            "n_frames": len(ans),
            "coverage": len(ans) / max(1, len(part)),
            "IR": ans["IR"].mean(),
            f"I@{delta}": ans[f"I@{delta}"].mean(),
            "frac_straddle": float((ans["attribution"] == "straddle").mean()),
            "frac_served_minor": float((ans["attribution"] == "served_minor").mean()),
        })
    return pd.DataFrame(rows).sort_values("n_modes")
