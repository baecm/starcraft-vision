# src/dataset/loader.py
from __future__ import annotations

from typing import Optional, Tuple, Iterable, List

import torch
from torch.utils.data import DataLoader, Subset

import src.utils as utils
from src.dataset.custom_penn_fudan import CustomPennFudanDataset
from src.dataset.splits import train_val_split_indices, subsample_indices
from src.utils.logger import Logger


def make_loader(
    ds,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> Optional[DataLoader]:
    """
    공통 DataLoader 생성 유틸.

    Args:
        ds: torch Dataset (또는 Subset)
        batch_size: 배치 크기
        shuffle: 셔플 여부
        num_workers: DataLoader worker 수

    Returns:
        DataLoader 또는 ds가 None이면 None
    """
    if ds is None:
        return None

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=utils.collate_fn,
    )


def unwrap_subset(ds):
    """
    torch.utils.data.Subset이 여러 겹으로 감싸져 있을 때
    최하단의 원본 dataset을 꺼내오는 유틸.
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
) -> Tuple[DataLoader, Optional[DataLoader], object]:
    """
    학습/검증용 DataLoader를 구성하는 유틸.

    - train_replays 로 train dataset을 구성
    - val_replays 가 주어지면, 그 replay들에서만 validation dataset을 구성
      (val_count 만큼 샘플; val_count <= 0 이면 전체 사용)
    - val_replays 가 없으면, train dataset 에서 랜덤 split 으로 val_count 만큼을 val로 사용
    - sample_ratio 는 train 에만 적용
    - 반환값:
        (train_loader, val_loader, inner_dataset)
        - inner_dataset 은 Subset 이 벗겨진 실제 CustomPennFudanDataset 인스턴스
          (in_channels 계산 등에 사용)

    Args:
        input_root: input/dst 루트
        label_root: label/dst 루트
        label_method: 라벨 메서드 이름
        window_size: 윈도우 크기
        interval: 프레임 샘플링 간격
        batch_size: 배치 크기
        num_workers: DataLoader workers
        train_replays: 학습에 사용할 replay ID 목록
        val_replays: (선택) 검증에 사용할 replay ID 목록 (없으면 train에서 split)
        sample_ratio: 학습 데이터 서브샘플링 비율 (0 < r <= 1)
        include_components: 사용할 components 리스트
        val_count: validation 샘플 개수 (0이면 val 없음)
        seed: (선택) 인덱스 셔플/샘플링용 시드
    """
    Logger.info("[Stage] Loading data...")
    Logger.info(f"[Info] Input root: {input_root}")
    Logger.info(f"[Info] Label root: {label_root}, method: {label_method}")

    train_ids: List[str] = [str(r) for r in train_replays]
    if not train_ids:
        raise ValueError("at least one train replay id must be provided (train_replays is empty).")
    Logger.info(f"[Info] Train IDs: {train_ids}")

    val_ids: List[str] = [str(r) for r in val_replays] if val_replays is not None else []
    if val_ids:
        Logger.info(f"[Info] Val/Test IDs: {val_ids}")
    else:
        Logger.info("[Info] Val/Test IDs not provided; will split from train set.")

    # --- Train용 full dataset 구성 ---
    train_full = CustomPennFudanDataset(
        input_root,
        label_root,
        label_method,
        training_ids=train_ids,
        training=True,
        window_size=window_size,
        interval=interval,
        include_components=include_components,
    )

    n_train_full = len(train_full)
    Logger.info(f"[Info] Full train dataset size: {n_train_full}")
    Logger.info(f"[Info] Window size: {window_size}, Interval: {interval}")

    # --- Validation dataset 구성 ---
    if val_ids:
        # 별도의 val_replays 에서 검증용 dataset 생성
        val_full = CustomPennFudanDataset(
            input_root,
            label_root,
            label_method,
            training_ids=val_ids,
            training=True,  # 기존 transform 스타일을 그대로 사용
            window_size=window_size,
            interval=interval,
            include_components=include_components,
        )
        n_val_full = len(val_full)
        Logger.info(f"[Info] Full val dataset size (from val_replays): {n_val_full}")

        if val_count > 0 and n_val_full > val_count:
            # val_full 에서 val_count 개수만큼만 랜덤 샘플
            _, val_idx = train_val_split_indices(
                n_samples=n_val_full,
                val_count=val_count,
                seed=seed,
            )
            val_dataset = Subset(val_full, val_idx)
            Logger.info(
                f"[Info] Sampled val dataset from val_replays: {len(val_dataset)} "
                f"(val_count={val_count})"
            )
        elif val_count <= 0:
            val_dataset = val_full
            Logger.info("[Info] Using all val_replays for validation (val_count <= 0).")
        else:
            val_dataset = val_full
            Logger.info(
                f"[Info] Using all val_replays for validation "
                f"(val_count={val_count} >= n_val_full={n_val_full})."
            )

        train_dataset = train_full

    else:
        # val_replays 가 없으면 train_full 에서 split
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
                f"[Info] Split into Train {len(train_dataset)} / Val {len(val_dataset)} "
                f"(val_count={val_count}) from train_replays."
            )
        else:
            if val_count <= 0:
                Logger.info("[Info] No validation split (val_count <= 0).")
            else:
                Logger.info(
                    f"[Info] No validation split from train_replays "
                    f"(val_count={val_count} >= n_train_full={n_train_full})."
                )

    # --- Train 서브샘플링 (sample_ratio) ---
    if sample_ratio < 1.0 and len(train_dataset) > 0:
        base_indices = list(range(len(train_dataset)))
        sampled_idx = subsample_indices(
            base_indices,
            ratio=sample_ratio,
            seed=seed,
        )
        train_dataset = Subset(train_dataset, sampled_idx)
        Logger.info(
            f"[Info] Applied sampling to train data (ratio={sample_ratio}): "
            f"Train {len(train_dataset)}"
        )

    # --- DataLoader 생성 ---
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = (
        make_loader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        if val_dataset is not None
        else None
    )

    # in_channels 계산용 inner dataset (Subset 벗겨진 원본)
    inner_dataset = unwrap_subset(train_dataset)

    return train_loader, val_loader, inner_dataset


__all__ = ["make_loader", "unwrap_subset", "load_data"]
