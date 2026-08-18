# src/model/plugins/kbrs/eval.py
from __future__ import annotations
import torch
from .scorer import make_ones_kernel, make_center_kernel

def evaluate_kbrs_score(
    score_map: torch.Tensor,
    components: dict[str, torch.Tensor]
) -> dict[str, float]:
    """
    Utility function to summarize KBRS scores for logging.
    """
    res = {}
    if score_map is not None:
        res["score_mean"] = float(score_map.mean().item())
        res["score_max"] = float(score_map.max().item())
    
    for k, v in components.items():
        if isinstance(v, torch.Tensor):
            res[f"{k}_mean"] = float(v.mean().item())
            res[f"{k}_max"] = float(v.max().item())
            
    return res
