"""Region of Common Interest (ROCI) target augmentation.

Joo et al. (2023) train their Mask R-CNN observer with an extra loss term over
viewports placed where the human observers' own viewports overlap. Their
Algorithm 1 builds those viewports from the ground truth alone - no model
output is involved - so this is label augmentation rather than a change to the
model, and it is applied here, once, when the labels are generated.

The procedure is the one in their paper: sum the observer viewports into a
coverage map, smooth it, keep the local maxima that clear a floor, drop any
peak too close to a stronger one, and take the strongest few. That is the same
sequence `metrics.modes.extract_modes` runs when it recovers the ranked modes
of the same distribution; it is reimplemented here rather than imported
because `metrics.modes` pulls in pycocotools and pandas, which the
preprocessing container has no other reason to carry.

What the two do with the peaks is where they part. ROCI hands them to the
detector as additional objects to find, which sharpens a single prediction
toward the consensus. Director-CenterNet ranks them by how many observers each
one represents and assigns them to separate output regions.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


def _coverage_map(
    viewports: Sequence[Tuple[int, int]],
    height: int,
    width: int,
    box_wh: Tuple[int, int],
) -> np.ndarray:
    """How many observer viewports cover each tile."""
    w, h = box_wh
    coverage = np.zeros((height, width), dtype=float)
    for x, y in viewports:
        x0 = int(np.clip(x, 0, width))
        y0 = int(np.clip(y, 0, height))
        x1 = int(np.clip(x + w, 0, width))
        y1 = int(np.clip(y + h, 0, height))
        if x1 > x0 and y1 > y0:
            coverage[y0:y1, x0:x1] += 1.0
    return coverage



@lru_cache(maxsize=None)
def _single_observer_peak(
    height: int,
    width: int,
    box_wh: Tuple[int, int],
    sigma: float,
) -> float:
    """What one observer's viewport is worth after the same smoothing.

    Joo et al. read their floor off the raw scale of the coverage map, where a
    tile one observer watches scores 1 and a tile two watch concurrently
    scores 2, and put the floor just above 1. Their Gaussian is a window the
    size of a viewport, so smoothing roughly preserves that scale and the
    number means what it says.

    Smoothing here is scipy's sigma form, which attenuates a peak by an amount
    that depends on sigma, on the viewport size and on the grid - so the raw
    number does not survive the change of filter. Measuring one observer under
    the very same filter restores the intent: the floor becomes a multiple of
    one observer's worth rather than an absolute that has to be recalibrated,
    and silently ends up admitting everything or nothing.

    The box is placed at the map centre so the reference is the untruncated
    peak; a viewport at the edge smooths lower under constant padding, and
    calibrating against that would make the floor depend on where the
    observers happened to be looking.
    """
    w, h = box_wh
    x = max(0, (width - w) // 2)
    y = max(0, (height - h) // 2)
    one = _coverage_map([(x, y)], height, width, box_wh)
    return float(gaussian_filter(one, sigma=sigma, mode="constant").max())

def roci_viewports(
    viewports: Sequence[Tuple[int, int]],
    height: int,
    width: int,
    box_wh: Tuple[int, int],
    sigma: float,
    min_sep: float,
    threshold: float,
    max_regions: int,
) -> List[Tuple[int, int]]:
    """Extra viewports for the regions the observers agree on.

    Returns top-left (x, y) pairs in the same convention as the observer
    labels, so the caller can append them to a frame's list and let the COCO
    export treat them like any other viewport.

    `threshold` is a multiple of one observer's worth, measured under the same
    filter by `_single_observer_peak`, so Joo et al.'s 1.1 keeps its meaning -
    just above what a lone observer produces - whatever sigma, viewport size or
    grid this pipeline uses. Without such a floor the local maxima of a frame
    where two observers overlap in one place and three others each look
    somewhere alone would yield five regions, four of which no two observers
    share; the floor is what makes the output common interest rather than every
    attention peak.

    An empty list comes back when no two observers overlap, which is Joo et
    al.'s `else` branch: with no common interest there is nothing to add, and
    the loss term is simply absent for that frame.
    """
    if len(viewports) < 2 or max_regions <= 0:
        return []

    coverage = _coverage_map(viewports, height, width, box_wh)
    # "if overlapping area exists" - a tile covered by one observer alone is
    # not common interest, so there is nothing to find unless two agree.
    if not np.any(coverage >= 2.0):
        return []

    smoothed = gaussian_filter(coverage, sigma=sigma, mode="constant")
    if smoothed.max() <= 0:
        return []

    # The floor is a multiple of what one observer is worth under this very
    # filter, not an absolute on the smoothed scale - see
    # `_single_observer_peak` for why the raw number does not survive the
    # change of Gaussian.
    reference = _single_observer_peak(height, width, tuple(box_wh), float(sigma))
    if reference <= 0:
        return []

    peaks = smoothed >= maximum_filter(smoothed, size=3, mode="constant")
    peaks &= smoothed >= threshold * reference
    rows, cols = np.nonzero(peaks)
    if len(rows) == 0:
        return []

    order = np.argsort(-smoothed[rows, cols])
    w, h = box_wh
    kept: List[Tuple[int, int]] = []
    centers: List[np.ndarray] = []
    for idx in order:
        centre = np.array([float(rows[idx]), float(cols[idx])])
        if any(np.linalg.norm(centre - k) < min_sep for k in centers):
            continue
        centers.append(centre)
        x = int(round(centre[1] - w / 2.0))
        y = int(round(centre[0] - h / 2.0))
        # A viewport is a camera rectangle and cannot hang off the map, so it
        # is shifted back inside rather than cropped - the same treatment the
        # repulsion loss gives a region at the boundary.
        x = int(np.clip(x, 0, max(0, width - w)))
        y = int(np.clip(y, 0, max(0, height - h)))
        kept.append((x, y))
        if len(kept) >= max_regions:
            break

    return kept
