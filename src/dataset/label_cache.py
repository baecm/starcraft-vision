# src/dataset/label_cache.py
from __future__ import annotations

import json
import os
import pickle
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from typing import Iterable, List

import tqdm

from utils.logger import Logger


def _process_json_worker(args: tuple[str, str, str]) -> str:
    """
    Worker for COCO JSON → pickle 변환.

    Args:
        args: (replay_id, label_root, label_method)

    Returns:
        상태 메시지 문자열
    """
    rid, label_root, label_method = args
    json_path = os.path.join(label_root, f"{rid}.rep", f"{label_method}.json")
    pkl_path = os.path.join(label_root, f"{rid}.rep", f"{label_method}.pkl")

    if not os.path.exists(json_path):
        return f"Skipped {rid}: no JSON found."
    if os.path.exists(pkl_path):
        return f"Skipped {rid}: pickle already exists."

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # 최소 필드 보정
        coco.setdefault("info", {"description": "auto-generated", "version": "1.0"})
        coco.setdefault("licenses", [])
        coco.setdefault("categories", [{"id": 1, "name": "viewport"}])
        coco.setdefault("images", [])
        coco.setdefault("annotations", [])

        payload = {
            "info": coco["info"],
            "licenses": coco["licenses"],
            "categories": coco["categories"],
            "images": coco["images"],
            "annotations": coco["annotations"],
        }

        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f)

        return f"Success {rid}: pickle created."
    except Exception as e:
        return f"Failed {rid}: {e}"


def ensure_label_pickles(
    label_root: str,
    label_method: str,
    replay_ids: Iterable[str | int],
    num_workers: int = 4,
    verbose: bool = True,
) -> None:
    """
    COCO JSON 라벨에서 pickle 캐시를 생성/보장하는 유틸.

    - JSON 파일이 없으면: "Skipped ...: no JSON found."
    - pickle 이 이미 있으면: "Skipped ...: pickle already exists."
    - 둘 다 아니면: JSON → pickle 변환

    Args:
        label_root: label/dst 루트 디렉토리
        label_method: 라벨링 메서드 이름 (예: "legacy", "kbrs", ...)
        replay_ids: 처리할 리플레이 ID 리스트
        num_workers: multiprocessing Pool 프로세스 개수
        verbose: Logger 에 진행 상황을 남길지 여부
    """
    def log(msg: str) -> None:
        if verbose:
            Logger.info(f"[LabelCache] {msg}")

    replay_ids_list: List[str] = [str(r) for r in replay_ids]
    if not replay_ids_list:
        log("No replay IDs given; nothing to do.")
        return

    log(
        f"Ensuring pickle cache for {len(replay_ids_list)} replays "
        f"(label_root={label_root}, method={label_method}, workers={num_workers})"
    )

    tasks = [(rid, label_root, label_method) for rid in replay_ids_list]

    # with Pool(processes=max(1, num_workers)) as pool:
    #     results = list(
    #         tqdm.tqdm(
    #             pool.imap_unordered(_process_json_worker, tasks),
    #             total=len(tasks),
    #             desc="Building label pickle cache",
    #         )
    #     )
    
    with ThreadPool(processes=max(1, num_workers)) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap_unordered(_process_json_worker, tasks),
                total=len(tasks),
                desc="Building label pickle cache",
            )
        )

    # 통계 집계
    success_count = sum(1 for r in results if r.startswith("Success"))
    skipped_exist_count = sum(1 for r in results if "pickle already exists" in r)
    skipped_no_json_count = sum(1 for r in results if "no JSON found" in r)
    failed_count = sum(1 for r in results if r.startswith("Failed"))

    log(
        "Label pickle cache summary. "
        f"Success: {success_count}, "
        f"Skipped (existing): {skipped_exist_count}, "
        f"Skipped (no JSON): {skipped_no_json_count}, "
        f"Failed: {failed_count}"
    )

    if failed_count > 0:
        for r in results:
            if r.startswith("Failed"):
                Logger.error(f"[LabelCache] {r}")


__all__ = ["ensure_label_pickles"]
