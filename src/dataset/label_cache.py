from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from typing import Iterable, List

import tqdm

from utils.logger import Logger

def _process_json_worker(args: tuple[str, str, str]) -> str:
    """
    Worker: convert one replay's COCO JSON labels to a pickle.
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

        # ========================================================
        # Number the observers of each frame: annotator_id 0..U-1
        # ========================================================
        raw_annotations = coco.get("annotations", [])
        grouped_by_image = defaultdict(list)
        
        for ann in raw_annotations:
            grouped_by_image[ann["image_id"]].append(ann)
            
        processed_annotations = []
        for image_id, anns in grouped_by_image.items():
            # the JSON order is not guaranteed, so order by annotation id first
            anns = sorted(anns, key=lambda x: x.get("id", 0))
            
            for idx, ann in enumerate(anns):
                ann["annotator_id"] = idx  # 0, 1, 2, 3, 4
                processed_annotations.append(ann)
                
        # replace with the numbered annotations
        coco["annotations"] = processed_annotations
        # ========================================================

        # fill in the COCO fields a consumer may expect
        coco.setdefault("info", {"description": "auto-generated", "version": "1.0"})
        coco.setdefault("licenses", [])
        coco.setdefault("categories", [{"id": 1, "name": "viewport"}])
        coco.setdefault("images", [])
        
        payload = {
            "info": coco["info"],
            "licenses": coco["licenses"],
            "categories": coco["categories"],
            "images": coco["images"],
            "annotations": coco["annotations"],  # with annotator_id
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
    Make sure every replay has a pickle cache of its COCO JSON labels.

    - no JSON: "Skipped ...: no JSON found."
    - pickle already there: "Skipped ...: pickle already exists."
    - otherwise: convert the JSON to a pickle

    Args:
        label_root: the label/dst root
        label_method: label file name without extension (e.g. "all_correct")
        replay_ids: replays to process
        num_workers: multiprocessing Pool size
        verbose: log progress
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

    # The bar is drawn only on a terminal (disable=None); in a log file the
    # summary line below says the same in one line.
    with ThreadPool(processes=max(1, num_workers)) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap_unordered(_process_json_worker, tasks),
                total=len(tasks),
                desc="Building label pickle cache",
                disable=None,
            )
        )

    # summary counts
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
