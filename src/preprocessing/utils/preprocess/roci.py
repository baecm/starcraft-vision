"""Region of Common Interest (ROCI) target augmentation.

Joo et al. (2023) train their Mask R-CNN observer with extra targets placed
where the human observers' own viewports overlap. Their Algorithm 1 derives
those viewports from the ground truth alone - no model output is involved - so
this is label augmentation rather than a change to the model, and it is applied
here, once, when the labels are generated.

This follows their released code rather than the paper's prose, since the two
differ in details that decide the outcome. Their pipeline is: sum the observer
masks, blur with a Gaussian the size of a viewport, zero everything at or below
1.1, and take the local maxima at least 7 tiles apart. Each surviving peak
becomes one more viewport centred on it, appended to the five human ones, which
is the "5 + n" target shape their Table 4 reports.

The blur is what makes the 1.1 mean anything. Their kernel is 19x11, about the
size of the 20x12 viewport itself, so a region two observers both cover still
reads near 2 after blurring, and a floor just above 1 separates common interest
from a lone observer. A wider blur flattens the scale and the same number
admits nothing. cv2 is not a dependency here, so the filter is reproduced with
scipy: the sigmas cv2 derives from a kernel size, truncated at that size, over
the reflect-101 border cv2 uses by default.

The same sequence recovers the ranked modes in `metrics.modes.extract_modes`.
What differs is what each does with the peaks: ROCI hands them to the detector
as more objects to find, sharpening one prediction toward the consensus, while
Director-CenterNet ranks them by observer support and gives them separate
output regions.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter


def _cv2_sigma(ksize: int) -> float:
    """The sigma cv2.GaussianBlur derives from a kernel size when given 0."""
    return 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8


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


def _blur_like_cv2(coverage: np.ndarray, ksize_wh: Tuple[int, int]) -> np.ndarray:
    """cv2.GaussianBlur(src, ksize, 0) with scipy.

    cv2 takes its kernel as (width, height), derives a sigma per axis from each
    size, cuts the kernel off at that size, and reflects at the border about
    the edge sample without repeating it. scipy's `mirror` is that same border,
    and `truncate` is expressed in sigmas, so the cutoff has to be converted.
    Separate 1-D passes are what allow a different truncation per axis.
    """
    k_w, k_h = ksize_wh
    sigma_x, sigma_y = _cv2_sigma(k_w), _cv2_sigma(k_h)
    radius_x, radius_y = (k_w - 1) // 2, (k_h - 1) // 2

    out = gaussian_filter1d(
        coverage, sigma=sigma_y, axis=0, mode="mirror",
        truncate=radius_y / sigma_y,
    )
    return gaussian_filter1d(
        out, sigma=sigma_x, axis=1, mode="mirror",
        truncate=radius_x / sigma_x,
    )


def _peak_local_max(image: np.ndarray, min_distance: int) -> np.ndarray:
    """skimage.feature.peak_local_max(image, min_distance) with scipy.

    A pixel is a peak when it equals the maximum over the (2d+1) square around
    it and lies above the image minimum. skimage also drops anything within
    `min_distance` of the border by default, which matters on a map this small.
    Returns (row, col) pairs.
    """
    size = 2 * min_distance + 1
    peaks = image == maximum_filter(image, size=size, mode="constant")
    peaks &= image > image.min()

    if min_distance > 0:
        peaks[:min_distance, :] = False
        peaks[-min_distance:, :] = False
        peaks[:, :min_distance] = False
        peaks[:, -min_distance:] = False

    rows, cols = np.nonzero(peaks)
    return np.stack([rows, cols], axis=1)


def roci_viewports(
    viewports: Sequence[Tuple[int, int]],
    height: int,
    width: int,
    box_wh: Tuple[int, int],
    ksize_wh: Tuple[int, int],
    min_distance: int,
    threshold: float,
    max_regions: int = 0,
) -> List[Tuple[int, int]]:
    """Extra viewports for the regions the observers agree on.

    Returns top-left (x, y) pairs in the observer labels' own convention, so a
    caller can append them to a frame's list and let the COCO export treat them
    like any other viewport. `max_regions` of 0 leaves the count unbounded,
    which is what Joo et al. do; their per-frame target count varies with how
    many regions of common interest the frame turned out to have.

    An empty list comes back when nothing clears the floor, which is their
    `else` branch: no common interest, so no extra term for that frame.
    """
    if len(viewports) < 2:
        return []

    coverage = _coverage_map(viewports, height, width, box_wh)
    blurred = _blur_like_cv2(coverage, ksize_wh)
    blurred = np.where(blurred <= threshold, 0.0, blurred)
    if not blurred.any():
        return []

    peaks = _peak_local_max(blurred, min_distance)
    if len(peaks) == 0:
        return []

    # strongest first, so a cap - if one is set - keeps the best rather than
    # whatever numpy happened to list first
    order = np.argsort(-blurred[peaks[:, 0], peaks[:, 1]])
    peaks = peaks[order]
    if max_regions > 0:
        peaks = peaks[:max_regions]

    w, h = box_wh
    result: List[Tuple[int, int]] = []
    for row, col in peaks:
        x = int(col) - w // 2
        y = int(row) - h // 2
        # Joo et al. slice the mask without clipping, so a peak near the left
        # or top edge produces a negative start, which numpy reads from the far
        # side and leaves the mask empty - an all-zero target that teaches
        # nothing. The viewport is shifted back inside instead, which is also
        # how the rest of this project treats a camera at the boundary.
        x = int(np.clip(x, 0, max(0, width - w)))
        y = int(np.clip(y, 0, max(0, height - h)))
        result.append((x, y))

    return result
