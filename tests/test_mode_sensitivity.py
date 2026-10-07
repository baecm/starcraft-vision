# tests/test_mode_sensitivity.py
#
# scripts/mode_sensitivity.py keeps its own copy of the second half of
# metrics.modes.extract_modes (so the smoothing runs once per sigma). This
# checks that the copy still ranks modes the same way. Runs under pytest or
# directly (`python3 tests/test_mode_sensitivity.py`).
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from metrics.modes import box_mask, extract_modes  # noqa: E402
from mode_sensitivity import ranked_support  # noqa: E402

H = W = 128
VH, VW = 12, 20


def _random_viewports(rng, n_obs=5):
    # observers clustered around a few points, as real frames are
    anchors = rng.uniform([0, 0], [W - VW, H - VH], size=(rng.integers(1, 4), 2))
    picks = anchors[rng.integers(0, len(anchors), n_obs)] + rng.normal(0, 6, (n_obs, 2))
    xy = np.clip(np.round(picks), 0, [W - VW, H - VH])
    return np.array([[x, y, VW, VH] for x, y in xy], dtype=float)


def test_copy_matches_extract_modes():
    rng = np.random.default_rng(0)
    for _ in range(300):
        boxes = _random_viewports(rng)
        for sigma, theta, sep in [(4.0, 0.35, 12.0), (2.0, 0.25, 8.0), (6.0, 0.5, 20.0)]:
            expected = extract_modes(boxes, H, W, sigma, sep, theta, 5).support.tolist()
            masks = [box_mask(b, H, W) for b in boxes]
            cov = np.zeros((H, W))
            for m in masks:
                cov += m
            cov /= max(1, len(masks))
            smoothed = gaussian_filter(cov, sigma=sigma, mode="constant")
            got = ranked_support(masks, smoothed, theta, sep, 5) if smoothed.max() > 0 else []
            assert [int(s) for s in got] == expected, (boxes, sigma, theta, sep, got, expected)


if __name__ == "__main__":
    test_copy_matches_extract_modes()
    print("ok  test_copy_matches_extract_modes")
