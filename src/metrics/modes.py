# src/metrics/modes.py
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
      directly. `IR` reuses `custom_evaluator.intersection_ratio` (the same
      function backing the project's `ic@000/030/050` numbers) instead of a
      second, independently maintained overlap formula.

Loading follows the rest of `src/metrics` and `src/evaluate.py`: ground truth
goes through `pycocotools.coco.COCO`, and predictions are grouped by
`image_id` the same way `evaluate.coco_to_kernel_labels` and
`multi_region_eval.compute_multi_region_metrics` already do.
`scripts/mode_disagreement.py` resolves replay/model names to actual file
paths via `evaluate.load_coco_gt` / `evaluate.load_coco_preds` (re-exported
by `estimate.py` for backwards compatibility; same
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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from scipy.ndimage import gaussian_filter, maximum_filter

from .custom_evaluator import intersection_ratio
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


def _boxes_and_scores(dets: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    items = sorted(dets, key=lambda a: -float(a.get("score", 1.0)))
    boxes = np.array([a["bbox"] for a in items], dtype=float)
    scores = np.array([float(a.get("score", 1.0)) for a in items], dtype=float)
    return boxes, scores


def predictions_from_dets(dets_by_img: Dict[int, List[dict]]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Adapt the {image_id: [det, ...]} grouping returned by
    `estimate.load_coco_preds` into the (boxes, scores) shape `analyse_method`
    expects, highest score first."""
    return {image_id: _boxes_and_scores(dets) for image_id, dets in dets_by_img.items() if dets}


def load_predictions(path: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """frame -> (boxes, scores), loaded directly from a prediction json file.

    Mirrors the preds_by_img grouping already used by
    `evaluate.coco_to_kernel_labels` and
    `multi_region_eval.compute_multi_region_metrics`. Prefer
    `estimate.load_coco_preds` + `predictions_from_dets` when the file lives
    at the project's usual `predictions/<model>/model_<epoch>/<replay>.rep/`
    layout, since that also handles score-threshold suffixed folders.
    """
    with open(path) as fh:
        data = json.load(fh)
    anns = data["annotations"] if isinstance(data, dict) else data

    by_img: Dict[int, List[dict]] = {}
    for a in anns:
        by_img.setdefault(int(a["image_id"]), []).append(a)
    return predictions_from_dets(by_img)


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


def _rle(mask: np.ndarray) -> dict:
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))


def coverage_of(box: np.ndarray, mask: np.ndarray, height: int, width: int) -> float:
    """|box ∩ mask| / |box|, i.e. the project's Intersection Ratio with `mask`
    as the target. Delegates to `custom_evaluator.intersection_ratio` (the
    same function behind `ic@000/030/050`) instead of reimplementing the
    ratio, so "IR" means the same thing everywhere in `src/metrics`."""
    return intersection_ratio(_rle(box_mask(box, height, width)), _rle(mask), denom="pred")


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
            coverage_of(primary, box_mask(box_from_center(c, size_hw, height, width), height, width),
                        height, width)
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
) -> pd.DataFrame:
    frames = sorted(set(gt_by_frame) & set(pred_by_frame))
    if not frames:
        raise ValueError("ground truth and prediction share no frame ids")

    span = max(frames) - min(frames) + 1
    records = []

    total_frames = len(frames)
    # same cadence as evaluator.eval_intersection_run / MultiRegionEvaluator:
    # at most ~10 updates per run, never more often than every 5000 frames,
    # so a fold-wide log file doesn't balloon on large replays.
    step_interval = max(5000, total_frames // 10)
    tag = f" {label}" if label else ""

    for idx, frame in enumerate(frames):
        if (idx + 1) % step_interval == 0 or (idx + 1) == total_frames:
            print(f"  [ModeDisagreement]{tag} {idx + 1}/{total_frames} frames "
                  f"({(idx + 1) / total_frames * 100:.1f}%)", flush=True)

        obs = gt_by_frame[frame]
        modes = extract_modes(
            obs, height, width,
            sigma=sigma, min_sep=min_sep,
            rel_threshold=rel_threshold, max_modes=max_modes,
        )

        boxes, scores = pred_by_frame[frame]
        if len(boxes) == 0:
            continue
        primary = boxes[0]

        ir = coverage_of(primary, modes.union, height, width)
        attribution, nearest, mode_cov = attribute(
            primary, modes, size_hw, height, width, delta, straddle_floor
        )

        row = {
            "frame": frame,
            "progression": (frame - min(frames)) / max(1, span - 1),
            "n_modes": len(modes.centers),
            "support_top1": int(modes.support[0]) if len(modes.support) else 0,
            "support_top2": int(modes.support[1]) if len(modes.support) > 1 else 0,
            "n_pred": len(boxes),
            "primary_score": float(scores[0]) if len(scores) else np.nan,
            "IR": ir,
            f"I@{delta}": float(ir >= delta),
            "attribution": attribution,
            "nearest_mode_rank": nearest,
            "best_mode_coverage": mode_cov,
            "primary_cy": box_center(primary)[0],
            "primary_cx": box_center(primary)[1],
        }
        row["margin"] = row["support_top1"] - row["support_top2"]

        for k in range(1, k_max + 1):
            top_k = boxes[:k]
            row[f"BoK{k}@{delta}"] = float(
                any(coverage_of(b, modes.union, height, width) >= delta for b in top_k)
            )
            served = 0
            for mask in modes.observers:
                if not mask.any():
                    continue
                # fraction of this observer's own viewport covered by box b,
                # i.e. intersection_ratio(b, mask, denom="gt")
                best = max(
                    (
                        intersection_ratio(_rle(box_mask(b, height, width)), _rle(mask), denom="gt")
                        for b in top_k
                    ),
                    default=0.0,
                )
                served += int(best >= delta)
            row[f"OC{k}@{delta}"] = served / max(1, modes.n_observers)

        records.append(row)

    df = pd.DataFrame(records).sort_values("frame").reset_index(drop=True)

    # temporal quantities, only across genuinely consecutive frames (a gap in
    # frame ids must not silently count as one "step" - see module docstring
    # on why this stays a local diff rather than a call into
    # evaluator.compute_m_cti, which has no such gap awareness)
    step = df["frame"].diff()
    disp = np.hypot(df["primary_cy"].diff(), df["primary_cx"].diff())
    df["VD_step"] = disp.where(step == step.mode().iloc[0] if len(step.mode()) else False)

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
) -> Dict[str, float]:
    """Whole-sequence velocity/jerk/jump_rate of the primary (top-1)
    prediction, via `evaluator.compute_m_cti` - the project's existing
    magnitude/threshold-based instability metric. Reported next to
    `mode_flip_rate`/`top2_flip_rate` for comparison, not as a replacement:
    see the module docstring for why the two are not interchangeable.
    """
    boxes_xyxy = []
    for f in frames:
        boxes, _ = pred_by_frame.get(f, (np.empty((0, 4)), np.empty(0)))
        if len(boxes) == 0:
            continue
        x, y, w, h = boxes[0]
        boxes_xyxy.append([x, y, x + w, y + h])

    if len(boxes_xyxy) < 2:
        return {"m_cti": 0.0, "jerk": 0.0, "jump_rate": 0.0, "velocity": 0.0}

    trajectory = np.array(boxes_xyxy, dtype=float)[:, None, :]  # (T, 1, 4)
    return compute_m_cti(trajectory, jump_threshold=jump_threshold, box_format="xyxy")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def summarise(df: pd.DataFrame, delta: float, k_max: int) -> dict:
    out = {
        "frames": len(df),
        "IR": df["IR"].mean(),
        f"I@{delta}": df[f"I@{delta}"].mean(),
        "VD": df["VD_step"].mean(),
        "mode_flip_rate": df["mode_flip"].mean(),
        "top2_flip_rate": df["top2_flip"].mean(),
        "mean_n_modes": df["n_modes"].mean(),
    }
    for k in range(1, k_max + 1):
        out[f"BoK{k}@{delta}"] = df[f"BoK{k}@{delta}"].mean()
        out[f"OC{k}@{delta}"] = df[f"OC{k}@{delta}"].mean()
    for label in ATTRIBUTIONS:
        out[f"frac_{label}"] = float((df["attribution"] == label).mean())
    misses = df[df[f"I@{delta}"] == 0]
    out["miss_rate"] = float(len(misses)) / max(1, len(df))
    for label in ATTRIBUTIONS:
        out[f"miss_{label}"] = (
            float((misses["attribution"] == label).mean()) if len(misses) else np.nan
        )
    return out


def by_quartile(df: pd.DataFrame, delta: float, k_max: int) -> pd.DataFrame:
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
    for margin, part in df.groupby("margin", observed=True):
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
    for replay, part in df.groupby("replay", observed=True):
        rows.append({
            "replay": replay,
            "n_frames": len(part),
            "IR": part["IR"].mean(),
            f"I@{delta}": part[f"I@{delta}"].mean(),
            "mode_flip_rate": part["mode_flip"].mean(),
            "top2_flip_rate": part["top2_flip"].mean(),
            "frac_straddle": float((part["attribution"] == "straddle").mean()),
            "frac_served_minor": float((part["attribution"] == "served_minor").mean()),
        })
    return pd.DataFrame(rows).sort_values("replay")


def by_n_modes(df: pd.DataFrame, delta: float) -> pd.DataFrame:
    rows = []
    for n, part in df.groupby("n_modes", observed=True):
        rows.append({
            "n_modes": n,
            "n_frames": len(part),
            "IR": part["IR"].mean(),
            f"I@{delta}": part[f"I@{delta}"].mean(),
            "frac_straddle": float((part["attribution"] == "straddle").mean()),
            "frac_served_minor": float((part["attribution"] == "served_minor").mean()),
        })
    return pd.DataFrame(rows).sort_values("n_modes")
