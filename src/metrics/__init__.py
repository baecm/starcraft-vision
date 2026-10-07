"""Evaluation metrics.

  custom_evaluator  single-region IR and Intersection@{any,0.3,0.5} (evaluate.py)
  evaluator         multi-region metrics: CWO, M-CTI, pairwise overlap (evaluate.py)
  modes             ranked attention modes and the per-frame mode analysis (scripts/)
"""
from .evaluator import (
    MultiRegionEvaluator,
    compute_cwo,
    compute_m_cti,
    compute_event_recall,
    compute_pairwise_overlap,
)
from .custom_evaluator import (
    ImageIR,
    eval_intersection_run,
    intersection_ratio,
)
from .modes import (
    Modes,
    analyse_method,
    attribute,
    by_margin,
    by_n_modes,
    by_quartile,
    by_replay,
    extract_modes,
    gt_boxes_by_frame,
    image_size,
    load_gt,
    load_predictions,
    predictions_from_dets,
    primary_track_m_cti,
    summarise as summarise_modes,
)

__all__ = [
    "MultiRegionEvaluator",
    "compute_cwo",
    "compute_m_cti",
    "compute_event_recall",
    "compute_pairwise_overlap",
    "ImageIR",
    "eval_intersection_run",
    "intersection_ratio",
    "Modes",
    "analyse_method",
    "attribute",
    "by_margin",
    "by_n_modes",
    "by_quartile",
    "by_replay",
    "extract_modes",
    "gt_boxes_by_frame",
    "image_size",
    "load_gt",
    "load_predictions",
    "predictions_from_dets",
    "primary_track_m_cti",
    "summarise_modes",
]
