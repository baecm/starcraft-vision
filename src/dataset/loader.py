"""DataLoaders for training: the train set, an optional validation subset, and their seeded splits."""
from __future__ import annotations

from typing import Optional, Tuple, Iterable, List

import torch
from torch.utils.data import DataLoader, Subset

import utils
from dataset.starcraft_windows import StarCraftWindowDataset
from dataset.splits import train_val_split_indices, subsample_indices
from utils.logger import Logger


from utils.seed import seed_worker


def make_loader(
    ds,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: Optional[int] = None,
) -> Optional[DataLoader]:
    """
    A DataLoader with seeded shuffling and worker initialization when seed is given.
    """
    if ds is None:
        return None

    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(int(seed))

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=utils.collate_fn,
        worker_init_fn=seed_worker if seed is not None else None,
        generator=g,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
    )


def unwrap_subset(ds):
    """
    The dataset under any number of nested torch.utils.data.Subset wrappers.
    """
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds


def load_data(
    input_root: str,
    label_root: str,
    label_method: str,
    window_size: int,
    interval: int,
    batch_size: int,
    num_workers: int,
    train_replays: Iterable[str | int],
    *,
    val_replays: Optional[Iterable[str | int]] = None,
    sample_ratio: float = 1.0,
    include_components: Optional[list[str]] = None,
    val_count: int = 1000,
    seed: Optional[int] = None,
    pair_mode: bool = False,
    use_mode_cache: bool = True,
    mode_targets: str = "none",
) -> Tuple[DataLoader, Optional[DataLoader], object]:
    """
    Build the training loader and, if asked for, a validation loader.

    - The training set is every window of train_replays.
    - With val_replays, validation is val_count windows drawn (seeded) from
      those replays (all of them if val_count <= 0); otherwise val_count
      windows are split off the training set.
    - sample_ratio subsamples the training set only.

    Returns (train_loader, val_loader or None, the StarCraftWindowDataset under
    any Subset wrappers). All draws use `seed`, so the same seed gives the
    same split, subsample and shuffle order.
    """
    Logger.info("[Stage] Loading data...")
    Logger.info(f"[Data] Input root: {input_root}")
    Logger.info(f"[Data] Label root: {label_root}, method: {label_method}")

    train_ids: List[str] = [str(r) for r in train_replays]
    if not train_ids:
        raise ValueError("at least one train replay id must be provided (train_replays is empty).")
    Logger.info(f"[Data] Train IDs: {train_ids}")

    val_ids: List[str] = [str(r) for r in val_replays] if val_replays is not None else []
    if val_ids:
        Logger.info(f"[Data] Val/Test IDs: {val_ids}")
    else:
        Logger.info("[Data] Val/Test IDs not provided; will split from train set.")

    # --- full training dataset ---
    train_full = StarCraftWindowDataset(
        input_root,
        label_root,
        label_method,
        training_ids=train_ids,
        training=True,
        window_size=window_size,
        interval=interval,
        include_components=include_components,
        pair_mode=pair_mode,
        use_mode_cache=use_mode_cache,
        mode_targets=mode_targets,
    )

    n_train_full = len(train_full)
    Logger.info(f"[Data] Full train dataset size: {n_train_full}")
    Logger.info(f"[Data] Window size: {window_size}, Interval: {interval}")

    # --- validation dataset ---
    if val_ids:
        # from the separate val_replays
        val_full = StarCraftWindowDataset(
            input_root,
            label_root,
            label_method,
            training_ids=val_ids,
            training=True,
            window_size=window_size,
            interval=interval,
            include_components=include_components,
            verbose=False,
            pair_mode=False,  # validation needs no consecutive pairs
            use_mode_cache=use_mode_cache,
            mode_targets=mode_targets,
        )
        n_val_full = len(val_full)
        Logger.info(f"[Data] Full val dataset size (from val_replays): {n_val_full}")

        if val_count > 0 and n_val_full > val_count:
            # a seeded random subset of val_count windows
            _, val_idx = train_val_split_indices(
                n_samples=n_val_full,
                val_count=val_count,
                seed=seed,
            )
            val_dataset = Subset(val_full, val_idx)
            Logger.info(
                f"[Data] Sampled val dataset from val_replays: {len(val_dataset)} "
                f"(val_count={val_count})"
            )
        elif val_count <= 0:
            val_dataset = val_full
            Logger.info("[Data] Using all val_replays for validation (val_count <= 0).")
        else:
            val_dataset = val_full
            Logger.info(
                f"[Data] Using all val_replays for validation "
                f"(val_count={val_count} >= n_val_full={n_val_full})."
            )

        train_dataset = train_full

    else:
        # no val_replays: split val_count windows off the training set
        train_dataset = train_full
        val_dataset = None

        if val_count and n_train_full > val_count:
            train_idx, val_idx = train_val_split_indices(
                n_samples=n_train_full,
                val_count=val_count,
                seed=seed,
            )
            train_dataset = Subset(train_full, train_idx)
            val_dataset = Subset(train_full, val_idx)
            Logger.info(
                f"[Data] Split into Train {len(train_dataset)} / Val {len(val_dataset)} "
                f"(val_count={val_count}) from train_replays."
            )
        else:
            if val_count <= 0:
                Logger.info("[Data] No validation split (val_count <= 0).")
            else:
                Logger.info(
                    f"[Data] No validation split from train_replays "
                    f"(val_count={val_count} >= n_train_full={n_train_full})."
                )

    # --- subsample the training set (sample_ratio) ---
    if sample_ratio < 1.0 and len(train_dataset) > 0:
        base_indices = list(range(len(train_dataset)))
        sampled_idx = subsample_indices(
            base_indices,
            ratio=sample_ratio,
            seed=seed,
        )
        train_dataset = Subset(train_dataset, sampled_idx)
        Logger.info(
            f"[Data] Applied sampling to train data (ratio={sample_ratio}): "
            f"Train {len(train_dataset)}"
        )

    # --- loaders ---
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
    )
    val_loader = (
        make_loader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=seed,
        )
        if val_dataset is not None
        else None
    )

    # the underlying dataset, for reading channel counts
    inner_dataset = unwrap_subset(train_dataset)

    return train_loader, val_loader, inner_dataset


__all__ = ["make_loader", "unwrap_subset", "load_data"]
