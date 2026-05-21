import os
import json
import argparse
import numpy as np
from collections import defaultdict
import multiprocessing as mp

from tqdm import tqdm
import sys

from src.utils.logger import Logger  

def build_ann_index(label_coco: dict) -> dict[int, list[dict]]:
    ann_by_image_id = defaultdict(list)
    for ann in label_coco.get("annotations", []):
        image_id = int(ann["image_id"])
        ann_by_image_id[image_id].append(ann)
    return ann_by_image_id


def coord_cast(xs: np.ndarray, ys: np.ndarray, mode: str):
    if mode == "int":
        return xs.astype(np.int64), ys.astype(np.int64)
    if mode == "round":
        return np.rint(xs).astype(np.int64), np.rint(ys).astype(np.int64)
    if mode == "floor":
        return np.floor(xs).astype(np.int64), np.floor(ys).astype(np.int64)
    if mode == "ceil":
        return np.ceil(xs).astype(np.int64), np.ceil(ys).astype(np.int64)
    raise ValueError(f"Unknown coord mode: {mode}")


def sample_channels_at_xy(npz, xy, clip_to_bounds: bool, coord_mode: str):
    if not xy:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty

    xs, ys = zip(*xy)
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)

    xs, ys = coord_cast(xs, ys, coord_mode)

    h, w = npz["density"].shape  # (H, W)
    if clip_to_bounds:
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)

    dens = npz["density"][ys, xs]
    cent = npz["centeredness"][ys, xs]
    mix = npz["mixture"][ys, xs]
    return dens, cent, mix


def parse_replays(values) -> list[int]:
    out = []
    for v in values:
        parts = str(v).split(",")
        for p in parts:
            p = p.strip()
            if not p:
                continue
            out.append(int(p))
    return sorted(set(out))


def parse_id_strings(values) -> list[str]:
    """
    supports:
      --id-strings a b c
      --id-strings a,b,c
    """
    if not values:
        return []
    out = []
    for v in values:
        parts = str(v).split(",")
        for p in parts:
            p = p.strip()
            if p:
                out.append(p)
    # dedup preserving order
    seen = set()
    uniq = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def resolve_npz_dir_for_replay(args, replay_id: int) -> str:
    # cache: {data_root}/{cache_subdir}/{replay_id}.rep/{image_id}.npz
    return os.path.join(args.data_root, args.cache_subdir, f"{replay_id}.rep")


def resolve_label_json_gt(args, replay_id: int) -> str:
    return os.path.join(
        args.data_root,
        args.label_subdir,
        f"{replay_id}.rep",
        f"{args.label_method}.json",
    )


def resolve_label_json_pred(args, replay_id: int, id_string: str) -> str:
    # pred: {prediction_path}/{id_string}/model_{epoch}/{replay_id}.rep/{label_method}.json
    return os.path.join(
        args.prediction_path,
        id_string,
        f"model_{int(args.epoch):03d}",
        f"{replay_id}.rep",
        f"{args.label_method}.json",
    )


def resolve_result_dir_gt(args) -> str:
    # gt: {result_root}/gt/{label_method}
    return os.path.join(args.result_root, "gt", args.label_method)


def resolve_result_dir_pred(args, id_string: str) -> str:
    # pred: {result_root}/pred/{id_string}/model_{epoch}/{label_method}
    return os.path.join(
        args.result_root,
        "pred",
        id_string,
        f"model_{int(args.epoch):03d}",
        args.label_method,
    )


def out_path_for_replay_gt(args, replay_id: int) -> str:
    out_dir = resolve_result_dir_gt(args)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"kbrs_samples_{replay_id}.json")


def out_path_for_replay_pred(args, replay_id: int, id_string: str) -> str:
    out_dir = resolve_result_dir_pred(args, id_string)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"kbrs_samples_{replay_id}.json")


# -------------------------
# Multiprocessing workers
# -------------------------

def _lookup_worker(task_q, result_q, worker_args: dict):
    Logger.set_level(worker_args["log_level"])

    replay_id = worker_args["replay_id"]
    npz_dir = worker_args["npz_dir"]
    clip = worker_args["clip"]
    coord_mode = worker_args["coord_mode"]
    skip_missing_npz = worker_args["skip_missing_npz"]
    force_row_on_error = worker_args["force_row_on_error"]

    while True:
        task = task_q.get()
        if task is None:
            break

        image_id, num_anns, xy = task
        npz_path = os.path.join(npz_dir, f"{image_id}.npz")

        base_row = {
            "replay_id": int(replay_id),
            "image_id": int(image_id),
            "num_anns": int(num_anns),
            "xy": xy,
        }

        try:
            if not xy:
                base_row.update(
                    {
                        "npz_found": bool(os.path.exists(npz_path)),
                        "density": [],
                        "centeredness": [],
                        "mixture": [],
                    }
                )
                result_q.put(base_row)
                continue

            if not os.path.exists(npz_path):
                if skip_missing_npz or force_row_on_error:
                    base_row.update(
                        {
                            "npz_found": False,
                            "density": [],
                            "centeredness": [],
                            "mixture": [],
                            "error": f"NPZ missing: {npz_path}",
                        }
                    )
                    result_q.put(base_row)
                    continue
                result_q.put(("__FATAL__", f"NPZ not found: {npz_path}"))
                continue

            npz = np.load(npz_path, allow_pickle=True)
            try:
                dens, cent, mix = sample_channels_at_xy(
                    npz, xy, clip_to_bounds=clip, coord_mode=coord_mode
                )
            finally:
                npz.close()

            base_row.update(
                {
                    "npz_found": True,
                    "density": dens.tolist(),
                    "centeredness": cent.tolist(),
                    "mixture": mix.tolist(),
                }
            )
            result_q.put(base_row)

        except Exception as e:
            if force_row_on_error:
                base_row.update(
                    {
                        "npz_found": bool(os.path.exists(npz_path)),
                        "density": [],
                        "centeredness": [],
                        "mixture": [],
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                result_q.put(base_row)
            else:
                result_q.put(("__FATAL__", f"frame={image_id} error: {type(e).__name__}: {e}"))


def _writer_worker(result_q, out_path: str, total_tasks: int, log_level: str, use_tqdm: bool):
    Logger.set_level(log_level)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True

        pbar = None
        if use_tqdm:
            pbar = tqdm(
                total=total_tasks,
                desc=f"write {os.path.basename(out_path)}",
                dynamic_ncols=True,
                leave=True,
                mininterval=2.0,
                miniters=200,
            )
        written = 0
        while written < total_tasks:
            msg = result_q.get()

            if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "__FATAL__":
                if pbar:
                    pbar.close()
                Logger.error(msg[1])
                raise RuntimeError(msg[1])

            row = msg

            if not first:
                f.write(",\n")
            else:
                first = False

            json.dump(row, f, ensure_ascii=False)
            written += 1

            if written % 1000 == 0:
                Logger.info(f"progress {written}/{total_tasks}")

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        f.write("\n]\n")


# -------------------------
# CLI / main
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Sample KBRS cache channels (density/centeredness/mixture) at bbox (x,y) for all frames in replays."
    )

    p.add_argument(
        "--data-root",
        type=str,
        default=os.path.join(os.getcwd(), "data"),
        help="Root data directory",
    )

    p.add_argument(
        "--replays",
        type=str,
        nargs="+",
        required=True,
        help="Replay IDs. Examples: --replays 275 438 OR --replays 275,438",
    )

    p.add_argument("--label-method", type=str, default="all_correct")

    p.add_argument(
        "--cache-subdir",
        type=str,
        default=os.path.join("input", "dst", "__kbrs_cache__"),
    )
    p.add_argument(
        "--label-subdir",
        type=str,
        default=os.path.join("label", "dst"),
        help="Used only when --label-source gt",
    )

    p.add_argument(
        "--label-source",
        type=str,
        default="gt",
        choices=["gt", "pred"],
        help="Where to read labels from (gt or pred)",
    )

    # pred settings (epoch is shared; id_strings can be multiple)
    p.add_argument(
        "--prediction-path",
        type=str,
        default=os.path.join(os.getcwd(), "predictions"),
        help="Prediction root dir (required if --label-source pred)",
    )
    p.add_argument(
        "--id-strings",
        type=str,
        nargs="*",
        default=[],
        help='Model id_strings (pred only). Examples: --id-strings a b OR --id-strings "a,b"',
    )
    p.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch number (pred only). Directory name is model_{epoch}",
    )

    p.add_argument("--clip", action="store_true")
    p.add_argument("--skip-missing-npz", action="store_true")
    p.add_argument(
        "--coord-mode",
        type=str,
        default="int",
        choices=["int", "round", "floor", "ceil"],
    )

    p.add_argument("--only-frames-with-anns", action="store_true")

    p.add_argument(
        "--result-root",
        type=str,
        default=os.path.join(os.getcwd(), "results", "kbrs_scores"),
        help="Root directory for results. Layout depends on --label-source.",
    )

    p.add_argument(
        "--log-level",
        type=str,
        default="log",
        choices=["none", "log", "debug"],
    )

    p.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars")

    p.add_argument(
        "--num-workers",
        type=int,
        default=min(os.cpu_count() or 4, 8),
        help="Number of lookup workers (default: min(cpu_count, 8))",
    )
    p.add_argument(
        "--queue-size",
        type=int,
        default=256,
        help="Max size for task/result queues (default: 256)",
    )
    p.add_argument(
        "--force-row-on-error",
        action="store_true",
        help="Always emit a row even when a frame fails (recommended for lossless output).",
    )

    return p.parse_args()


def _build_tasks(args, label_coco: dict) -> tuple[list[tuple[int, int, list[tuple[float, float]]]], int]:
    ann_by_image_id = build_ann_index(label_coco)

    if "images" in label_coco and label_coco["images"]:
        image_ids = sorted(int(img["id"]) for img in label_coco["images"])
        Logger.info("frames =", len(image_ids), "(from images)")
    else:
        image_ids = sorted(ann_by_image_id.keys())
        Logger.warn("label_coco['images'] missing/empty. frames =", len(image_ids), "(from annotations)")

    tasks = []
    skipped_no_anns = 0

    for image_id in image_ids:
        anns = ann_by_image_id.get(image_id, [])

        if args.only_frames_with_anns and len(anns) == 0:
            skipped_no_anns += 1
            continue

        xy = []
        for ann in anns:
            bb = ann.get("bbox", None)
            if not bb or len(bb) < 2:
                continue
            xy.append((bb[0], bb[1]))

        tasks.append((int(image_id), int(len(anns)), xy))

    return tasks, skipped_no_anns


def _run_replay_with_label_json(args, replay_id: int, label_json: str, out_path: str):
    npz_dir = resolve_npz_dir_for_replay(args, replay_id)

    Logger.debug("label_json =", label_json)
    Logger.debug("npz_dir     =", npz_dir)
    Logger.info("out_path =", out_path)

    if not os.path.exists(label_json):
        Logger.error("Label JSON not found:", label_json)
        raise FileNotFoundError(f"Label JSON not found: {label_json}")

    with open(label_json, "r", encoding="utf-8") as f:
        label_coco = json.load(f)

    tasks, skipped_no_anns = _build_tasks(args, label_coco)
    total_tasks = len(tasks)
    Logger.info("tasks =", total_tasks, "skipped_no_anns =", skipped_no_anns)

    ctx = mp.get_context("spawn")
    task_q = ctx.Queue(maxsize=args.queue_size)
    result_q = ctx.Queue(maxsize=args.queue_size)

    worker_args = {
        "replay_id": int(replay_id),
        "npz_dir": npz_dir,
        "clip": bool(args.clip),
        "coord_mode": args.coord_mode,
        "skip_missing_npz": bool(args.skip_missing_npz),
        "force_row_on_error": bool(args.force_row_on_error) or bool(args.skip_missing_npz),
        "log_level": args.log_level,
    }
    
    use_tqdm = (not args.no_tqdm) and sys.stderr.isatty()
    writer = ctx.Process(
        target=_writer_worker,
        args=(result_q, out_path, total_tasks, args.log_level, use_tqdm),
        daemon=False,
    )
    writer.start()

    num_workers = max(1, int(args.num_workers))
    workers = []
    for _ in range(num_workers):
        p = ctx.Process(
            target=_lookup_worker,
            args=(task_q, result_q, worker_args),
            daemon=False,
        )
        p.start()
        workers.append(p)

    for t in tasks:
        task_q.put(t)

    for _ in workers:
        task_q.put(None)

    for p in workers:
        p.join()

    writer.join()


def run_one_replay_gt(args, replay_id: int):
    label_json = resolve_label_json_gt(args, replay_id)
    out_path = out_path_for_replay_gt(args, replay_id)

    Logger.info("-----")
    Logger.info("Replay:", replay_id, "| source=gt")
    _run_replay_with_label_json(args, replay_id, label_json, out_path)
    Logger.info(f"Done replay={replay_id} saved={out_path}")


def run_one_replay_pred(args, replay_id: int, id_string: str):
    label_json = resolve_label_json_pred(args, replay_id, id_string)
    out_path = out_path_for_replay_pred(args, replay_id, id_string)

    Logger.info("-----")
    Logger.info("Replay:", replay_id, "| source=pred | id_string =", id_string, "| epoch =", args.epoch)
    _run_replay_with_label_json(args, replay_id, label_json, out_path)
    Logger.info(f"Done replay={replay_id} id_string={id_string} saved={out_path}")


def main():
    args = parse_args()
    Logger.set_level(args.log_level)

    replay_ids = parse_replays(args.replays)

    Logger.info("Starting calculate_kernel_score")
    Logger.info("label_source =", args.label_source)
    Logger.info("replays =", replay_ids)
    Logger.info("label_method =", args.label_method)
    Logger.info("result_root =", args.result_root)

    os.makedirs(args.result_root, exist_ok=True)

    if args.label_source == "gt":
        for replay_id in replay_ids:
            run_one_replay_gt(args, replay_id)
        Logger.info("All done.")
        return

    # pred mode validations
    if not args.prediction_path:
        raise ValueError("--prediction-path is required when --label-source pred")
    if args.epoch is None:
        raise ValueError("--epoch is required when --label-source pred")
    id_strings = parse_id_strings(args.id_strings)
    if not id_strings:
        raise ValueError("--id-strings is required (non-empty) when --label-source pred")

    Logger.info("prediction_path =", args.prediction_path)
    Logger.info("epoch =", args.epoch)
    Logger.info("id_strings =", id_strings)

    # iterate models then replays (or swap order if you prefer)
    for id_string in id_strings:
        for replay_id in replay_ids:
            run_one_replay_pred(args, replay_id, id_string)
    Logger.info("All done.")


if __name__ == "__main__":
    main()
