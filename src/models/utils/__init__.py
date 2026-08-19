# src/model/utils/__init__.py
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou
from .feature_utils import (
    aux_boost_loss,
    aux_entropy_sharp,
    pick_feature_map,
    auto_expand_indices,
    normalize_projections,
    compute_gate_from_raw_inputs,
    reduce_map_stats
)
from .transforms import CustomRCNNTransform
