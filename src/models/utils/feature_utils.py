import torch
import torch.nn.functional as F

from collections import OrderedDict
from typing import Dict, List, Tuple
from omegaconf import OmegaConf, DictConfig, ListConfig

def aux_boost_loss(x, tau=1.0, norm="zscore", margin=None):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x.clamp(-1e3, 1e3)
    
    if norm == "zscore":
        m = x.mean(dim=(-2,-1), keepdim=True)
        s = x.std(dim=(-2,-1), keepdim=True).clamp_min(1e-6)
        x = (x - m) / s
    elif norm == "minmax":
        mn = x.amin(dim=(-2,-1), keepdim=True)
        mx = x.amax(dim=(-2,-1), keepdim=True)
        x = (x - mn) / (mx - mn + 1e-6)
        x = 2*x - 1  # [-1,1]

    x = x.clamp(-10.0, 10.0)
    if margin is not None:
        return torch.relu(margin - x).mean()
    return F.softplus(-tau * x).mean()

def aux_entropy_sharp(score_map):
    p = torch.sigmoid(score_map)
    return -torch.mean(p * torch.log(p + 1e-6))

def pick_feature_map(features: OrderedDict[str, torch.Tensor] | Dict[str, torch.Tensor], pref: str) -> Tuple[str, torch.Tensor]:
    if pref in features:
        return pref, features[pref]
    k = max(features.keys(), key=lambda kk: features[kk].shape[-2] * features[kk].shape[-1])
    return k, features[k]

def auto_expand_indices(indices: List[int], window_size: int, per_window: int, in_channels: int) -> List[int]:
    if not indices:
        return indices
    if window_size > 1 and max(indices) < per_window and in_channels == per_window * window_size:
        out = []
        for t in range(window_size):
            out.extend([base + t * per_window for base in indices])
        return sorted(list(set(out)))
    return indices

def normalize_projections(
    projections_cfg,
    window_size: int,
    per_window: int,
    in_channels: int
) -> Dict[str, List[int]]:
    if projections_cfg is None:
        return {}
    
    if isinstance(projections_cfg, (DictConfig, ListConfig)):
        projections_cfg = OmegaConf.to_container(projections_cfg, resolve=True)
    
    proj_dict: Dict[str, List[int]] = {}

    if isinstance(projections_cfg, dict):
        for name, chs in projections_cfg.items():
            proj_dict[name] = auto_expand_indices(list(chs), window_size, per_window, in_channels)
        return proj_dict

    if isinstance(projections_cfg, list):
        for item in projections_cfg:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            chs  = item.get("channels", [])
            if name is None:
                continue
            proj_dict[name] = auto_expand_indices(list(chs), window_size, per_window, in_channels)
        return proj_dict

    return {}

def compute_gate_from_raw_inputs(
    raw_images: List[torch.Tensor],
    ref_fmap: torch.Tensor,
    gate_channels: List[int],
    region_size: Tuple[int, int],
    score_stride: int,
    gate_gain: float,
    reduce: str = "max",
) -> torch.Tensor:
    if len(gate_channels) == 0:
        return torch.ones((ref_fmap.size(0), *ref_fmap.shape[-2:]), device=ref_fmap.device, dtype=ref_fmap.dtype)

    N = len(raw_images)
    H, W = raw_images[0].shape[-2:]

    x = torch.stack(raw_images, dim=0).to(device=ref_fmap.device, dtype=ref_fmap.dtype)
    sub = x[:, gate_channels, :, :]
    if reduce == "max":
        sub = sub.max(dim=1, keepdim=True).values
    else:
        sub = sub.mean(dim=1, keepdim=True)

    oh, ow = ref_fmap.shape[-2], ref_fmap.shape[-1]
    g = F.adaptive_avg_pool2d(sub, output_size=(oh, ow)).squeeze(1)

    return (1.0 + gate_gain * g)

def reduce_map_stats(comp_map: torch.Tensor) -> Dict[str, torch.Tensor]:
    N = comp_map.size(0)
    flat = comp_map.view(N, -1)
    return {
        "mean": flat.mean(dim=1),
        "max":  flat.max(dim=1).values,
        "sum":  flat.sum(dim=1),
    }
