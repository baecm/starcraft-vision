from __future__ import annotations

import os
import pickle
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import tqdm

import config
from metrics.modes import extract_modes
from utils.logger import Logger


def get_mode_cache_filename(
    label_method: str,
    sigma: float = config.MODE_EXTRACTION_SIGMA,
    min_sep: float = config.MODE_EXTRACTION_MIN_SEP,
    rel_threshold: float = config.MODE_EXTRACTION_REL_THRESHOLD,
    max_modes: int = config.MODE_EXTRACTION_MAX_MODES,
) -> str:
    """Deterministic cache filename encoding mode extraction hyperparameters."""
    return f"modes_{label_method}_sig{sigma:.1f}_sep{min_sep:.1f}_th{rel_threshold:.2f}_k{max_modes}.pkl"


def get_mode_cache_path(
    label_root: str,
    rid: str | int,
    label_method: str,
    sigma: float = config.MODE_EXTRACTION_SIGMA,
    min_sep: float = config.MODE_EXTRACTION_MIN_SEP,
    rel_threshold: float = config.MODE_EXTRACTION_REL_THRESHOLD,
    max_modes: int = config.MODE_EXTRACTION_MAX_MODES,
) -> str:
    fname = get_mode_cache_filename(label_method, sigma, min_sep, rel_threshold, max_modes)
    return os.path.join(label_root, f"{rid}.rep", fname)


def _load_gt_boxes(
    label_root: str,
    rid: str,
    label_method: str,
) -> Tuple[Dict[int, np.ndarray], int, int]:
    """Load frame -> obs_boxes (U, 4) from either .pkl or .json."""
    pkl_path = os.path.join(label_root, f"{rid}.rep", f"{label_method}.pkl")
    json_path = os.path.join(label_root, f"{rid}.rep", f"{label_method}.json")

    data = None
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    elif os.path.exists(json_path):
        import json
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise FileNotFoundError(f"Neither {pkl_path} nor {json_path} exists for replay {rid}.")

    images = data.get("images", [])
    if images:
        h = int(images[0].get("height", config.ORIGIN_SHAPE[0]))
        w = int(images[0].get("width", config.ORIGIN_SHAPE[1]))
    else:
        h, w = config.ORIGIN_SHAPE

    raw_anns = data.get("annotations", [])
    by_frame: Dict[int, List[list]] = {}
    for ann in raw_anns:
        fid = int(ann["image_id"])
        bbox = ann.get("bbox")
        if bbox is not None and len(bbox) == 4:
            by_frame.setdefault(fid, []).append(bbox)

    gt_by_frame: Dict[int, np.ndarray] = {
        fid: np.array(boxes, dtype=float) for fid, boxes in by_frame.items() if len(boxes) > 0
    }
    return gt_by_frame, h, w


def _process_mode_worker(args: Tuple[str, str, str, float, float, float, int]) -> str:
    rid, label_root, label_method, sigma, min_sep, rel_threshold, max_modes = args
    cache_path = get_mode_cache_path(
        label_root, rid, label_method, sigma, min_sep, rel_threshold, max_modes
    )

    if os.path.exists(cache_path):
        return f"Skipped {rid}: mode cache already exists."

    try:
        gt_by_frame, h, w = _load_gt_boxes(label_root, rid, label_method)
    except Exception as e:
        return f"Failed {rid}: {e}"

    if not gt_by_frame:
        return f"Skipped {rid}: no GT boxes found."

    cached_modes = {}
    for fid, obs_boxes in gt_by_frame.items():
        modes = extract_modes(
            obs_boxes=obs_boxes,
            height=h,
            width=w,
            sigma=sigma,
            min_sep=min_sep,
            rel_threshold=rel_threshold,
            max_modes=max_modes,
        )
        cached_modes[fid] = {
            "centers": modes.centers.astype(np.float32),  # (M, 2) [row, col]
            "support": modes.support.astype(np.int64),    # (M,)
            "n_observers": int(modes.n_observers),
        }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    temp_path = f"{cache_path}.tmp_{os.getpid()}"
    with open(temp_path, "wb") as f:
        pickle.dump(cached_modes, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_path, cache_path)

    return f"Success {rid}: mode cache created ({len(cached_modes)} frames)."


def ensure_mode_cache(
    label_root: str,
    label_method: str,
    replay_ids: Iterable[str | int],
    sigma: float = config.MODE_EXTRACTION_SIGMA,
    min_sep: float = config.MODE_EXTRACTION_MIN_SEP,
    rel_threshold: float = config.MODE_EXTRACTION_REL_THRESHOLD,
    max_modes: int = config.MODE_EXTRACTION_MAX_MODES,
    num_workers: int = 4,
    verbose: bool = True,
) -> None:
    """Precompute and cache ranked attention modes for all given replays."""
    def log(msg: str) -> None:
        if verbose:
            Logger.info(msg)

    rids = [str(r) for r in replay_ids]
    if not rids:
        log("[ModeCache] Replay IDs empty. Nothing to process.")
        return

    tasks = [
        (rid, label_root, label_method, sigma, min_sep, rel_threshold, max_modes)
        for rid in sorted(set(rids))
    ]

    log(
        f"[ModeCache] Checking mode caches for {len(tasks)} replays "
        f"(sigma={sigma}, min_sep={min_sep}, rel_threshold={rel_threshold}, max_modes={max_modes})..."
    )

    if num_workers > 1:
        try:
            with Pool(processes=num_workers) as pool:
                results = list(
                    tqdm.tqdm(
                        pool.imap_unordered(_process_mode_worker, tasks),
                        total=len(tasks),
                        desc="ModeCache",
                        disable=None if verbose else True,
                    )
                )
        except Exception:
            with ThreadPool(processes=num_workers) as pool:
                results = list(
                    tqdm.tqdm(
                        pool.imap_unordered(_process_mode_worker, tasks),
                        total=len(tasks),
                        desc="ModeCache (threaded)",
                        disable=None if verbose else True,
                    )
                )
    else:
        results = [_process_mode_worker(t) for t in tqdm.tqdm(tasks, desc="ModeCache (seq)", disable=None if verbose else True)]

    n_success = sum(1 for r in results if r.startswith("Success"))
    n_skip = sum(1 for r in results if r.startswith("Skipped"))
    n_fail = sum(1 for r in results if r.startswith("Failed"))
    log(f"[ModeCache] Done. Created={n_success}, Skipped={n_skip}, Failed={n_fail}")


def load_mode_cache(
    label_root: str,
    rid: str | int,
    label_method: str,
    sigma: float = config.MODE_EXTRACTION_SIGMA,
    min_sep: float = config.MODE_EXTRACTION_MIN_SEP,
    rel_threshold: float = config.MODE_EXTRACTION_REL_THRESHOLD,
    max_modes: int = config.MODE_EXTRACTION_MAX_MODES,
) -> Dict[int, dict]:
    """Load precomputed mode cache for a replay. Returns frame_id -> dict."""
    cache_path = get_mode_cache_path(
        label_root, rid, label_method, sigma, min_sep, rel_threshold, max_modes
    )
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(f"[ModeCache] Cache not found: {cache_path}. Run ensure_mode_cache() first.")
    with open(cache_path, "rb") as f:
        return pickle.load(f)
