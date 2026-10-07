from __future__ import annotations

from typing import List, Sequence, Tuple, Optional

import torch


def _make_generator(seed: Optional[int] = None) -> Optional[torch.Generator]:
    """
    A torch.Generator seeded with `seed`, or None (the global RNG) when seed is None.
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
    Split range(n_samples) into train and val indices, val_count of them for val.

    Args:
        n_samples: number of samples
        val_count: number of validation samples
        seed: optional seed

    Returns:
        (train_indices, val_indices)
    """
    if n_samples <= 0:
        return [], []

    # no validation set when val_count is 0 or covers everything
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
    A random subset of `indices` of size max(1, int(len * ratio)), in random order.

    Args:
        indices: the indices to sample from
        ratio: 0.0 < ratio <= 1.0
        seed: optional seed

    Returns:
        the sampled indices
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
