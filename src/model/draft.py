import torch
import torch.nn.functional as F
from typing import Callable, Dict, List, Tuple

# ---------- 공용 연산자들 (예시) ----------
def op_density(patches: torch.Tensor) -> torch.Tensor:
    """
    patches: (N, Cg, kh*kw, L)
    return:  (N, L)  # 채널+공간 합
    """
    return patches.sum(dim=(1, 2))

def op_mixture_any(patches: torch.Tensor) -> torch.Tensor:
    """
    윈도우 내 값 합이 >0 인 채널 수
    """
    ch_active = (patches.sum(dim=2) > 0).float()  # (N, Cg, L)
    return ch_active.sum(dim=1)                   # (N, L)

def make_op_centeredness(kh: int, kw: int, sigma_scale: float = 4.0) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    가우시안 가중합 연산자 팩토리
    """
    cy, cx = (kh - 1) / 2.0, (kw - 1) / 2.0
    sigma = kh / sigma_scale

    def fn(patches: torch.Tensor) -> torch.Tensor:
        # patches: (N, Cg, kh*kw, L)
        device = patches.device
        yy, xx = torch.meshgrid(
            torch.arange(kh, device=device),
            torch.arange(kw, device=device),
            indexing="ij"
        )
        w = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))  # (kh, kw)
        w = w.reshape(1, 1, kh * kw, 1)                                        # (1,1,kh*kw,1)
        return (patches * w).sum(dim=(1, 2))                                    # (N, L)

    return fn

# ---------- 메인: 채널 그룹별로 다른 연산을 적용 ----------
def kbrs_score_map_unfold_per_channel(
    x: torch.Tensor,                                   # (N, C, H, W)  ← (N,C,W,H)이면 미리 permute
    kernel_size: Tuple[int, int] = (20, 12),
    stride: int | Tuple[int, int] = 1,
    padding: int | Tuple[int, int] = 0,
    dilation: int | Tuple[int, int] = 1,
    groups: List[Dict] = None,
    default_group: Dict | None = None,                 # 지정 안 된 채널용 (선택)
) -> torch.Tensor:
    """
    groups: 각 원소는 {"channels": List[int], "fn": Callable, "weight": float}
      - channels: 이 그룹에 포함될 채널 인덱스들
      - fn: (patches_subset: (N, Cg, kh*kw, L)) -> (N, L) 로 축약하는 함수
      - weight: 스칼라 가중치

    default_group: {"fn": Callable, "weight": float}  # groups에 포함되지 않은 채널에 일괄 적용 (선택)

    return: score_map (N, out_h, out_w)
    """
    assert x.dim() == 4, "x must be (N,C,H,W)"
    N, C, H, W = x.shape
    kh, kw = kernel_size

    if isinstance(stride, int):   stride   = (stride, stride)
    if isinstance(padding, int):  padding  = (padding, padding)
    if isinstance(dilation, int): dilation = (dilation, dilation)

    # unfold → (N, C*kh*kw, L) → (N, C, kh*kw, L)
    cols = F.unfold(
        x, kernel_size=(kh, kw),
        dilation=dilation, padding=padding, stride=stride
    )
    L = cols.shape[-1]
    patches = cols.view(N, C, kh * kw, L)

    # 출력 맵 크기
    out_h = (H + 2*padding[0] - dilation[0]*(kh - 1) - 1)//stride[0] + 1
    out_w = (W + 2*padding[1] - dilation[1]*(kw - 1) - 1)//stride[1] + 1

    total = torch.zeros((N, L), device=x.device)

    used = torch.zeros(C, dtype=torch.bool, device=x.device)
    groups = groups or []

    # 1) 명시 그룹들 처리
    for g in groups:
        chs: List[int] = g["channels"]
        fn: Callable = g["fn"]
        w: float = float(g.get("weight", 1.0))
        if len(chs) == 0:
            continue
        # (N, Cg, kh*kw, L)
        p_sub = patches[:, chs, :, :]
        total += w * fn(p_sub)
        used[torch.tensor(chs, device=x.device, dtype=torch.long)] = True

    # 2) 나머지 채널에 default 연산 적용(옵션)
    if default_group is not None:
        rem_idx = torch.nonzero(~used, as_tuple=False).flatten()
        if rem_idx.numel() > 0:
            p_sub = patches.index_select(dim=1, index=rem_idx)  # (N, Cr, kh*kw, L)
            fn = default_group["fn"]
            w  = float(default_group.get("weight", 1.0))
            total += w * fn(p_sub)

    # (N, out_h, out_w)
    return total.view(N, out_h, out_w)
