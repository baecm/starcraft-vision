import os
import glob

import argparse
import json
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from enum import Enum
from numpy.lib.stride_tricks import sliding_window_view


class Channel(Enum):
    Player_1_Worker = 0
    Player_1_Ground = 1
    Player_1_Air = 2
    Player_1_Building = 3

    Player_2_Worker = 4
    Player_2_Ground = 5
    Player_2_Air = 6
    Player_2_Building = 7

    Resource = 8
    Vision = 9
    Terrain = 10


def build_json_paths(
    directory: str,
    label_method: str = "all_correct",
    replays: Optional[List[str]] = None,
) -> List[str]:
    directory = os.path.abspath(directory)
    if replays:
        return [os.path.join(directory, rep, f"{label_method}.json") for rep in replays]
    return [str(p) for p in Path(directory).glob("**/*.json")]


def read_one_json(path: str, use_orjson: bool = True) -> Tuple[str, Any, Optional[str]]:
    try:
        if use_orjson:
            try:
                import json

                with open(path, "rb") as f:
                    return path, json.loads(f.read()), None
            except ModuleNotFoundError:
                pass  # orjson이 없으면 표준 json로 폴백
        with open(path, "r", encoding="utf-8") as f:
            return path, json.load(f), None
    except Exception as e:
        return path, None, f"{type(e).__name__}: {e}"


def read_json_files_threaded(
    directory: str,
    label_method: str = "all_correct",
    replays: Optional[List[str]] = None,
    max_workers: Optional[int] = None,
) -> List[Any]:
    paths = build_json_paths(directory, label_method, replays)
    if not paths:
        return []

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(32, cpu * 5)

    results: List[Any] = []
    errors: List[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for path, data, err in ex.map(read_one_json, paths):
            if err:
                errors.append(f"[FAIL] {path} -> {err}")
            else:
                results.append()

    if errors:
        print("\n".join(errors))
    return results


def read_json_files_processes(
    directory: str,
    label_method: str = "all_correct",
    replays: Optional[List[str]] = None,
    max_workers: Optional[int] = None,
) -> List[Any]:
    paths = build_json_paths(directory, label_method, replays)
    if not paths:
        return []

    if max_workers is None:
        max_workers = os.cpu_count() or 4

    results: List[Any] = []
    errors: List[str] = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for path, data, err in ex.map(read_one_json, paths):
            if err:
                errors.append(f"[FAIL] {path} -> {err}")
            else:
                results.append(data)

    if errors:
        print("\n".join(errors))
    return results


def get_annotations(data, replay_suffix=".rep"):
    frames = []

    for item in data or []:
        anns = item.get("annotations", [])
        if not anns:
            continue

        df = pd.DataFrame(anns)

        desc = (item.get("info") or {}).get("description", "") or ""
        replay_id = None
        if isinstance(desc, str) and desc.strip():
            replay_id = desc.strip().split()[-1]

        if replay_id is None:
            replay_id = "unknown"

        if replay_suffix and not str(replay_id).endswith(replay_suffix):
            replay_id = f"{replay_id}{replay_suffix}"

        df["replay"] = replay_id
        frames.append(df)

    base_cols = [
        "replay",
        "id",
        "image_id",
        "category_id",
        "bbox",
        "area",
        "segmentation",
        "iscrowd",
    ]
    if not frames:
        return pd.DataFrame(columns=base_cols + ["score"])

    has_score = any("score" in f.columns for f in frames)

    wanted_cols = base_cols + (["score"] if has_score else [])
    for i, f in enumerate(frames):
        for col in wanted_cols:
            if col not in f.columns:
                frames[i][col] = pd.NA

    annotations = pd.concat(frames, ignore_index=True)

    return annotations[wanted_cols]


def set_human_id(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["hid"] = dataframe.groupby(["replay", "image_id"]).cumcount()
    wide_dataframe = dataframe.pivot(
        index=["replay", "image_id"], columns="hid", values="bbox"
    )
    wide_dataframe.columns = [f"gt_{i}" for i in wide_dataframe.columns]
    return wide_dataframe.reset_index(inplace=True)


def get_npy_paths(dataframe: pd.DataFrame, data_dir: str) -> List[np.ndarray]:
    npy_paths = [
        os.path.join(
            data_dir, "input", "dst", item["replay"], str(item["image_id"]) + ".npy"
        )
        for _, item in dataframe.iterrows()
    ]
    return npy_paths


def gaussian_kernel2d(kw, kh, sigma=None, dtype=np.float32):
    if sigma is None:
        sigma_y = max(kh, 1) / 3.0
        sigma_x = max(kw, 1) / 3.0
    elif np.isscalar(sigma):
        sigma_y = sigma_x = float(sigma)
    else:
        sigma_y, sigma_x = map(float, sigma)

    y = np.arange(kh, dtype=dtype) - (kh - 1) / 2.0
    x = np.arange(kw, dtype=dtype) - (kw - 1) / 2.0
    Y, X = np.meshgrid(y, x, indexing="ij")
    K = np.exp(-(Y**2) / (2 * sigma_y**2) - (X**2) / (2 * sigma_x**2)).astype(
        dtype, copy=False
    )
    K /= K.sum() + 1e-6
    return K  # (kh, kw), sum≈1


def get_projection(x: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    return x.sum(axis=0 if axis == 0 else 1, keepdims=keepdims)


def score_density(x: np.ndarray, kw: int, kh: int) -> np.ndarray:
    _, H, W = x.shape

    x_sum = x.sum(axis=0, keepdims=True).astype(np.float32, copy=False)  # (N,H,W)
    x_sum = x_sum.reshape(H, W).astype(np.float32, copy=False)  # (H,W)

    ii = np.pad(x_sum, ((1, 0), (1, 0)), mode="constant")
    ii = ii.cumsum(axis=0).cumsum(axis=1)

    win_sum = (
        ii[kh:, kw:]  # bottom-right
        - ii[:-kh, kw:]  # top-right
        - ii[kh:, :-kw]  # bottom-left
        + ii[:-kh, :-kw]  # top-left
    )

    density = win_sum / float(kh * kw)  # (oh,ow)
    return density


def score_centeredness(
    x: np.ndarray, kw: int, kh: int, stride: int = 1, sigma: float = None
) -> np.ndarray:
    _, H, W = x.shape

    x_sum = x.sum(axis=0, keepdims=True).astype(np.float32, copy=False)
    x_sum = x_sum.reshape(H, W).astype(np.float32, copy=False)  # (H,W)

    K = gaussian_kernel2d(kw, kh, sigma=sigma, dtype=np.float32)  # (kh, kw), sum=1
    denom = float(K.sum()) + 1e-6

    patches = sliding_window_view(x_sum, (kh, kw))  # (N, H-kh+1, W-kw+1, kh, kw)

    out = (patches * K).sum(axis=(-1, -2))  # (N, H-kh+1, W-kw+1)

    # 평균 정규화 (PyTorch 경로와 동일)
    denom = float(K.sum()) + 1e-6
    out = out / denom

    # 입력 dtype 정책: 부동소수면 입력 dtype으로 반환, 정수 입력이면 float32 유지
    if x.dtype.kind == "f":
        out = out.astype(x.dtype, copy=False)
    return out


def score_mixture(
    x: np.ndarray,
    kw: int,
    kh: int,
    eps: float = 1e-6,
    pow_k: float = 1.0,
    stride=1,
    mode="entropy",
) -> np.ndarray:

    # ONES = np.ones((kh, kw), dtype=np.float32)
    xA = x[0:4, ...].sum(axis=0, keepdims=False).astype(np.float32, copy=False)  # (H,W)
    xB = x[4:8, ...].sum(axis=0, keepdims=False).astype(np.float32, copy=False)  # (H,W)

    def _sum_pool2d(x1):
        ii = np.pad(x1, ((1, 0), (1, 0)), mode="constant")
        ii = ii.cumsum(axis=0).cumsum(axis=1)

        win_sum = (
            ii[kh:, kw:]  # bottom-right
            - ii[:-kh, kw:]  # top-right
            - ii[kh:, :-kw]  # bottom-left
            + ii[:-kh, :-kw]  # top-left
        )
        return win_sum

    A = _sum_pool2d(xA)  # (oh,ow)
    B = _sum_pool2d(xB)  # (oh,ow)

    den = np.maximum(A + B, eps, dtype=np.float32)  # (oh,ow)
    p = A / den  # (oh,ow)

    if mode == "entropy":
        p1 = np.clip(p, eps, 1.0 - eps)
        q1 = np.clip(1.0 - p, eps, 1.0 - eps)

        conf = -(p1 * np.log(p1) + q1 * np.log(q1)) / np.log(2.0)  # (oh,ow), [0,1]
    elif mode == "gini":
        conf = 2.0 * p * (1.0 - p)  # (oh,ow), [0,0.5]
    elif mode == "linear":
        conf = 1.0 - np.abs(2.0 * p - 1.0)  # (oh,ow), [0,1]
    else:
        conf = 4.0 * p * (1.0 - p)  # (oh,ow), [0,1]

    if pow_k != 1.0:
        conf = np.clip(conf, 0.0, 1.0) ** pow_k  # (oh,ow), [0,1]

    return conf, A, B


def overview(ax: plt.Axes, x: np.ndarray):
    # Terrain & resource
    ax.imshow(x[Channel.Terrain.value] > 0, cmap="Greys", alpha=0.5)
    resource_alpha = np.where(x[Channel.Resource.value] == 1, 1.0, 0.0)
    ax.imshow(x[Channel.Resource.value] == 1, cmap="BuGn", alpha=resource_alpha)

    # Player colors
    p1_cmap = "Greens"
    p2_cmap = "Reds"

    # Player 1 & 2 - worker
    player_1_worker_alpha = np.where(x[Channel.Player_1_Worker.value] != 0, 1.0, 0.0)
    player_2_worker_alpha = np.where(x[Channel.Player_2_Worker.value] != 0, 1.0, 0.0)
    ax.imshow(
        x[Channel.Player_1_Worker.value] != 0, cmap=p1_cmap, alpha=player_1_worker_alpha
    )
    ax.imshow(
        x[Channel.Player_2_Worker.value] != 0, cmap=p2_cmap, alpha=player_2_worker_alpha
    )

    # Player 1 & 2 - ground
    player_1_ground_alpha = np.where(x[Channel.Player_1_Ground.value] != 0, 1.0, 0.0)
    player_2_ground_alpha = np.where(x[Channel.Player_2_Ground.value] != 0, 1.0, 0.0)
    ax.imshow(
        x[Channel.Player_1_Ground.value] != 0, cmap=p1_cmap, alpha=player_1_ground_alpha
    )
    ax.imshow(
        x[Channel.Player_2_Ground.value] != 0, cmap=p2_cmap, alpha=player_2_ground_alpha
    )

    # Player 1 & 2 - air
    player_1_air_alpha = np.where(x[Channel.Player_1_Air.value] != 0, 1.0, 0.0)
    player_2_air_alpha = np.where(x[Channel.Player_2_Air.value] != 0, 1.0, 0.0)
    ax.imshow(
        x[Channel.Player_1_Air.value] != 0, cmap=p1_cmap, alpha=player_1_air_alpha
    )
    ax.imshow(
        x[Channel.Player_2_Air.value] != 0, cmap=p2_cmap, alpha=player_2_air_alpha
    )

    # Player 1 & 2 - building
    player_1_building_alpha = np.where(
        x[Channel.Player_1_Building.value] != 0, 1.0, 0.0
    )
    player_2_building_alpha = np.where(
        x[Channel.Player_2_Building.value] != 0, 1.0, 0.0
    )
    ax.imshow(
        x[Channel.Player_1_Building.value] != 0,
        cmap=p1_cmap,
        alpha=player_1_building_alpha,
    )
    ax.imshow(
        x[Channel.Player_2_Building.value] != 0,
        cmap=p2_cmap,
        alpha=player_2_building_alpha,
    )

    # Vision mask
    vision_alpha = np.where(x[Channel.Vision.value] == 1, 0.0, 0.9)
    ax.imshow(
        np.zeros_like(x[Channel.Vision.value]), cmap="Greys_r", alpha=vision_alpha
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", type=str, nargs="+", required=True)
    parser.add_argument(
        "--storage-dir", type=str, default="/mnt/nas/baecm/starcraft-vision"
    )
    parser.add_argument("--label-method", type=str, default="all_correct")
    parser.add_argument("--max-workers", type=int, default=os.cpu_count() // 2)
    parser.add_argument("--kernel-width", type=int, default=20)
    parser.add_argument("--kernel-height", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_arguments()
    storage_dir = Path(args.storage_dir)

    data_dir = os.path.join(storage_dir, "data")
    ground_truth_dir = os.path.join(data_dir, "label", "dst")

    replays = [str(r) + ".rep" for r in args.replays]

    ground_truth_json_path = [
        f
        for name in replays
        for f in glob.glob(os.path.join(ground_truth_dir, name, "*.json"))
    ]

    ground_truth_data = read_json_files_processes(
        ground_truth_dir,
        label_method=args.label_method,
        replays=replays,
        max_workers=args.max_workers,
    )

    annotation_ground_truth = get_annotations(ground_truth_data)

    wide_ground_truth = set_human_id(annotation_ground_truth)

    npy_paths = get_npy_paths(wide_ground_truth, data_dir)
    print(f"Total {len(npy_paths)} npy files found.")
