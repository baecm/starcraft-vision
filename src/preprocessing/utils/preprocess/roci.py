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

    `threshold` is absolute, on the smoothed coverage, as in Joo et al.: their
    delta sits just above 1 because 1 is a tile one observer watches and 2 a
    tile two watch concurrently, so the floor is what separates common interest
    from a lone observer. Note their Gaussian is a window of the viewport's own
    size while this uses scipy's sigma parameterisation, so the two attenuate
    peaks differently and the floor does not carry over numerically. Check the
    reported count per frame after a change: a floor set too high silently
    yields no regions at all and trains exactly like plain labels.

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

    peaks = smoothed >= maximum_filter(smoothed, size=3, mode="constant")
    peaks &= smoothed >= threshold
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
