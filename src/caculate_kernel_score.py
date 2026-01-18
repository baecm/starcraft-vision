import os
import json
import argparse
import numpy as np
from collections import defaultdict

from utils.logger import Logger  # src 기준 import 가정


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


def resolve_paths_for_replay(args, replay_id: int):
    data_root = args.data_root
    npz_dir = os.path.join(data_root, args.cache_subdir, f"{replay_id}.rep")
    label_json = os.path.join(
        data_root,
        args.label_subdir,
        f"{replay_id}.rep",
        f"{args.label_method}.json",
    )
    return label_json, npz_dir


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


def out_path_for_replay(result_root: str, replay_id: int, label_method: str) -> str:
    os.makedirs(result_root, exist_ok=True)
    return os.path.join(result_root, f"kbrs_samples_{replay_id}_{label_method}.json")


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
        help="Directory to write per-replay results: {result_root}/kbrs_samples_{replay_id}_{label_method}.json",
    )

    p.add_argument(
        "--log-level",
        type=str,
        default="log",
        choices=["none", "log", "debug"],
    )

    return p.parse_args()


def run_one_replay(args, replay_id: int):
    label_json, npz_dir = resolve_paths_for_replay(args, replay_id)

    Logger.info("-----")
    Logger.info("Replay:", replay_id)
    Logger.debug("label_json =", label_json)
    Logger.debug("npz_dir     =", npz_dir)

    if not os.path.exists(label_json):
        Logger.error("Label JSON not found:", label_json)
        raise FileNotFoundError(f"Label JSON not found: {label_json}")

    with open(label_json, "r", encoding="utf-8") as f:
        label_coco = json.load(f)

    ann_by_image_id = build_ann_index(label_coco)

    if "images" in label_coco and label_coco["images"]:
        image_ids = sorted(int(img["id"]) for img in label_coco["images"])
        Logger.info("frames =", len(image_ids), "(from images)")
    else:
        image_ids = sorted(ann_by_image_id.keys())
        Logger.warn(
            "label_coco['images'] missing/empty. frames =",
            len(image_ids),
            "(from annotations)",
        )

    rows = []
    missing_npz = 0
    processed = 0
    skipped_no_anns = 0

    for idx, image_id in enumerate(image_ids):
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

        npz_path = os.path.join(npz_dir, f"{image_id}.npz")

        Logger.debug(
            f"[{idx+1}/{len(image_ids)}] frame={image_id} anns={len(anns)} xy={len(xy)}"
        )

        if not os.path.exists(npz_path):
            missing_npz += 1
            Logger.warn("NPZ missing:", npz_path)

            if args.skip_missing_npz:
                rows.append(
                    {
                        "replay_id": int(replay_id),
                        "image_id": int(image_id),
                        "num_anns": int(len(anns)),
                        "xy": xy,
                        "npz_found": False,
                        "density": [],
                        "centeredness": [],
                        "mixture": [],
                    }
                )
                continue
            raise FileNotFoundError(f"NPZ not found: {npz_path}")

        npz = np.load(npz_path, allow_pickle=True)
        try:
            dens, cent, mix = sample_channels_at_xy(
                npz, xy, clip_to_bounds=args.clip, coord_mode=args.coord_mode
            )
        finally:
            npz.close()

        rows.append(
            {
                "replay_id": int(replay_id),
                "image_id": int(image_id),
                "num_anns": int(len(anns)),
                "xy": xy,
                "npz_found": True,
                "density": dens.tolist(),
                "centeredness": cent.tolist(),
                "mixture": mix.tolist(),
            }
        )
        processed += 1

        if (idx + 1) % 500 == 0:
            Logger.info(
                "Progress:",
                idx + 1,
                "/",
                len(image_ids),
                "processed=",
                processed,
                "missing_npz=",
                missing_npz,
            )

    Logger.info(
        f"Done replay={replay_id} frames_total={len(image_ids)} processed={processed} "
        f"missing_npz={missing_npz} skipped_no_anns={skipped_no_anns}"
    )
    return rows


def main():
    args = parse_args()
    Logger.set_level(args.log_level)

    replay_ids = parse_replays(args.replays)
    Logger.info("Starting calculate_kernel_score")
    Logger.info("replays =", replay_ids)
    Logger.info("label_method =", args.label_method)
    Logger.debug("data_root =", args.data_root)

    # result-root 없으면 저장 안 함 (원하면 required=True로 바꿔도 됨)
    if args.result_root:
        os.makedirs(args.result_root, exist_ok=True)
        Logger.info("result_root =", args.result_root)
    else:
        Logger.warn("result_root is empty -> will not write output files.")

    total_rows = 0

    for replay_id in replay_ids:
        rows = run_one_replay(args, replay_id)
        total_rows += len(rows)

        if args.result_root:
            out_path = out_path_for_replay(
                args.result_root, replay_id, args.label_method
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
            Logger.info("Saved:", out_path)

    Logger.info("All done. total_rows =", total_rows)


if __name__ == "__main__":
    main()
