# src/estimate.py
"""
Legacy entry point for replay-level estimation & evaluation.
All core metric calculations and CLI functionalities are now unified into src/evaluate.py.
"""
from __future__ import annotations

# Re-export all key functions from evaluate for full backwards compatibility
from evaluate import (
    main,
    parse_args,
    load_coco_gt,
    load_coco_preds,
    compute_ic_for_replay,
    compute_kbrs_for_replay,
    compute_multi_region_for_replay,
    make_gaussian_kernel,
    kbrs_scores_from_patch,
    extract_window,
    compute_kbrs_grid,
    eval_kernel_from_coco,
)

if __name__ == "__main__":
    main()
