# src/dataset/splits.py
from __future__ import annotations

from typing import List, Sequence, Tuple, Optional

import torch


def _make_generator(seed: Optional[int] = None) -> Optional[torch.Generator]:
    """
    선택적인 seed를 받아 torch.Generator를 생성.
    seed가 None이면 None을 반환해서 기본 전역 RNG를 사용하게 함.
    """
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def train_val_split_indices(
    n_samples: int,
    val_count: int,
    seed: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """
    전체 n_samples에서 val_count 만큼을 validation에 할당하는 인덱스 스플릿 함수.

    Args:
        n_samples: 전체 샘플 수
        val_count: validation으로 사용할 샘플 수
        seed: (선택) 랜덤 시드

    Returns:
        (train_indices, val_indices)
    """
    if n_samples <= 0:
        return [], []

    # val_count가 0이거나 전체 이상이면 val을 만들지 않음
    if val_count <= 0 or val_count >= n_samples:
        return list(range(n_samples)), []

    g = _make_generator(seed)
    if g is None:
        perm = torch.randperm(n_samples).tolist()
    else:
        perm = torch.randperm(n_samples, generator=g).tolist()

    val_idx = perm[-val_count:]
    train_idx = perm[:-val_count]

    return train_idx, val_idx


def subsample_indices(
    indices: Sequence[int],
    ratio: float,
    seed: Optional[int] = None,
) -> List[int]:
    """
    주어진 인덱스들에서 ratio 비율만큼 서브샘플링.

    Args:
        indices: 원본 인덱스 시퀀스
        ratio: 0.0 < ratio <= 1.0
        seed: (선택) 랜덤 시드

    Returns:
        서브샘플링된 인덱스 리스트
    """
    if not indices:
        return []

    if ratio >= 1.0:
        return list(indices)

    n = len(indices)
    keep_n = max(1, int(n * ratio))

    g = _make_generator(seed)
    idx_tensor = torch.as_tensor(indices, dtype=torch.long)

    if g is None:
        perm = idx_tensor[torch.randperm(n)]
    else:
        perm = idx_tensor[torch.randperm(n, generator=g)]

    return perm[:keep_n].tolist()


__all__ = ["train_val_split_indices", "subsample_indices"]
