# src/model/utils.py
import torch
import torch.nn.functional as F

from collections import OrderedDict
from typing import Dict, List, Tuple

def aux_boost_loss(x, tau=1.0, norm="zscore", margin=None):
    # ❶ 정규화로 동적 범위 맞추기
    if norm == "zscore":
        m = x.mean(dim=(-2,-1), keepdim=True)
        s = x.std(dim=(-2,-1), keepdim=True).clamp_min(1e-6)
        x = (x - m) / s
    elif norm == "minmax":
        mn = x.amin(dim=(-2,-1), keepdim=True)
        mx = x.amax(dim=(-2,-1), keepdim=True)
        x = (x - mn) / (mx - mn + 1e-6)
        x = 2*x - 1  # [-1,1]로 센터링

    # ❷ 포화에 덜 민감한 softplus 형태 (const 차이만 있음)
    if margin is not None:
        # x가 margin보다 크도록 밀어주는 힌지형 보조
        return torch.relu(margin - x).mean()
    return torch.nn.functional.softplus(-tau * x).mean()


def aux_entropy_sharp(score_map):
    p = torch.sigmoid(score_map)
    return -torch.mean(p * torch.log(p + 1e-6))


def pick_feature_map(features: "OrderedDict[str, torch.Tensor]", pref: str) -> Tuple[str, torch.Tensor]:
    if pref in features:
        return pref, features[pref]
    # fallback: 가장 큰 해상도
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
    """
    원본 입력(list of Tensor(Cin,H,W))에서 gate_channels 선택 → 채널 reduce(max/mean)
    → score_map 해상도(oh,ow)에 맞게 adaptive_avg_pool → (1 + gate_gain * g)
    """
    if len(gate_channels) == 0:
        return torch.ones((ref_fmap.size(0), *ref_fmap.shape[-2:]), device=ref_fmap.device, dtype=ref_fmap.dtype)

    N = len(raw_images)
    H, W = raw_images[0].shape[-2:]
    Cin = raw_images[0].shape[0]

    # stack
    x = torch.stack(raw_images, dim=0).to(device=ref_fmap.device, dtype=ref_fmap.dtype)  # (N,Cin,H,W)
    sub = x[:, gate_channels, :, :]  # (N,G,H,W)
    if reduce == "max":
        sub = sub.max(dim=1, keepdim=True).values
    else:
        sub = sub.mean(dim=1, keepdim=True)  # (N,1,H,W)

    # score_map 크기 = conv with stride on feature map, 하지만 여기선 간단히 adaptive로 맞춤
    # (실제로는 transform scale을 반영해 더 정확히 맵핑 할 수도 있음)
    # target size 추정: ref score_map과 동일해야 하므로, region/stride를 모르면 ref_fmap로부터 추정하기 어려움.
    # 여기서는 ref_fmap의 spatial 크기를 사용 → scorer가 같은 stride로 conv하므로 ok
    oh, ow = ref_fmap.shape[-2], ref_fmap.shape[-1]
    g = F.adaptive_avg_pool2d(sub, output_size=(oh, ow)).squeeze(1)  # (N,oh,ow)

    return (1.0 + gate_gain * g)  # (N,oh,ow)


def reduce_map_stats(comp_map: torch.Tensor) -> Dict[str, torch.Tensor]:
    N = comp_map.size(0)
    flat = comp_map.view(N, -1)
    return {
        "mean": flat.mean(dim=1),
        "max":  flat.max(dim=1).values,
        "sum":  flat.sum(dim=1),
    }