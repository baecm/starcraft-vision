#!/usr/bin/env python
# src/inference.py

import os
import sys
import tqdm
import multiprocessing
import json
import gc

import random
import secrets
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset

from cli import parse_inference_args
from dataset.inference_dataset import InferenceDataset

from model.factory import build_model
from types import SimpleNamespace
from omegaconf import OmegaConf  # KBRS YAML 로드를 위해 추가

import config
from utils.logger import Logger
from utils.synology_chat import send_message


from utils.seed import set_global_seed


def collate_fn(batch):
    """
    Bundle a batch into the format ([images], [metadata]).
    The metadata is a list of (replay_id, frame_id) tuples.
    """
    images, metas = zip(*batch)
    return list(images), list(metas)


def _available_cpu_count() -> int:
    """Estimate usable CPU cores (affinity-aware if possible)."""
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return multiprocessing.cpu_count()


def _auto_num_workers(device: torch.device) -> int:
    """
    Recommended DataLoader workers:
    - GPU: max(1, avail-1)  to keep I/O pipeline busy without oversubscription
    - CPU: max(0, avail-1)  to avoid contention with compute
    """
    avail = _available_cpu_count()
    if device.type == "cuda":
        return max(1, avail - 1)
    return max(0, avail - 1)


def _load_model(model_path: str, device: torch.device, in_channels: int, window_size: int, 
                architecture: str, num_classes: int = 2, use_kbrs: bool = False, 
                kbrs_params: dict = None, rtdetr_version: str = "v1", rtdetr_size: str = "l"):
    
    # 1. factory.py가 요구하는 인자들을 담을 dummy args(SimpleNamespace) 생성
    model_args = SimpleNamespace(
        model_name=architecture.lower(),
        use_kbrs=use_kbrs,
        num_classes=num_classes,
        in_channels=in_channels,
        window_size=window_size,
        kbrs_params=kbrs_params,
        loss_weights=None,
        rtdetr_version=rtdetr_version,  
        rtdetr_size=rtdetr_size
    )
    
    # 2. factory를 통해 모델 구조 생성 (Train과 완벽히 동일한 경로)
    model = build_model(model_args)
    
    # 3. 학습된 가중치 로드
    state = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    elif 'model' in state:
        state = state['model']
        
    missing, unexpected = model.load_state_dict(state, strict=False)
    
    if len(missing) > 0:
        Logger.warn(f"[Inference] Missing keys: {len(missing)} items")
    if len(unexpected) > 0:
        Logger.warn(f"[Inference] Unexpected keys: {len(unexpected)} items")
        
    model.to(device)
    model.eval()
    return model


def save_predictions_as_coco(
    replay_id: str,
    replay_results: list,
    label_method: str,
    output_dir: str
):
    """
    Save a single replay's predictions in COCO format (masks omitted for compactness).
    """
    out_dir = os.path.join(output_dir, f"{replay_id}.rep")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label_method}.json")

    categories = [{"id": 1, "name": "viewport", "supercategory": "viewport"}]

    coco = {
        "info": {
            "description": f"Predictions for replay {replay_id}",
            "version": "1.0",
            "label_method": label_method,
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    # prevent duplicate image entries
    seen_frames = set()
    ann_id = 1

    for item in replay_results:
        fid = int(item["frame_id"])

        # images: single per frame
        if fid not in seen_frames:
            coco["images"].append({
                "id": fid,
                "file_name": f"{replay_id}.rep/{fid}.npy",
                "width": int(config.ORIGIN_SHAPE[1]),
                "height": int(config.ORIGIN_SHAPE[0]),
            })
            seen_frames.add(fid)

        # annotations (bbox-only; polygon is derived from bbox for viewer compatibility)
        for box, score, label in zip(item["boxes"], item["scores"], item["labels"]):
            x1, y1, x2, y2 = map(int, box)
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            segmentation = [[x1, y1, x1 + w, y1, x1 + w, y1 + h, x1, y1 + h]]

            coco["annotations"].append({
                "id": ann_id,
                "image_id": fid,
                "category_id": int(label),   # single class -> 1 also OK
                "bbox": [x1, y1, w, h],
                "score": float(score),
                "area": int(w * h),
                "segmentation": segmentation,
                "iscrowd": 0,
            })
            ann_id += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    Logger.info(f"[Inference] Saved predictions for replay {replay_id} -> {out_path}")
    return out_path


def run_inference(args):
    Logger.info("[Inference] Starting...")

    if getattr(args, "seed", None) is None:
        generated = secrets.randbits(31)
        args.seed = generated
        Logger.info(f"[Seed] No --seed provided for inference; generated seed={generated}")
    else:
        Logger.info(f"[Seed] Using provided inference seed={args.seed}")
    set_global_seed(int(args.seed))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    Logger.info(f"[Inference] Using device: {device}")

    # Checkpoint path
    model_folder = os.path.join(args.model_root, args.model_name)
    model_number = int(getattr(args, "model_number"))
    model_path = os.path.join(model_folder, f"model_{model_number:03d}.pth")
    Logger.info(f"[Inference] Checkpoint path: {model_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # Decide output run dir
    if not getattr(args, "output_dir", None):   
        args.output_dir = "predictions"
        
    default_run_name = os.path.join(args.model_name, f"model_{model_number:03d}")
    run_name = getattr(args, "run_name", None) or default_run_name
    run_dir = os.path.join(args.output_dir, run_name)
    
    Logger.info(f"[Inference] Output run dir: {run_dir}")
    try:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "seed.txt"), "w", encoding="utf-8") as f:
            f.write(str(args.seed) + "\n")
    except Exception as e:
        Logger.warn(f"[Inference] Failed to write seed.txt: {e}")

    # Infer input channels once from a small temp dataset (first replay)
    temp_input_root = os.path.join(args.data_root, "input", "dst")
    temp_dataset = InferenceDataset(
        temp_input_root,
        [args.replays[0]],
        window_size=args.window_size,
        include_components=args.include_components
    )
    in_channels = len(temp_dataset.channel_indices) * temp_dataset.window_size
    del temp_dataset

    # 1. KBRS 파라미터 로드 (kbrs.yaml 연동)
    use_kbrs_flag = getattr(args, "use_kbrs", False)
    kbrs_params = None
    if use_kbrs_flag:
        try:
            yaml_path = os.path.join(os.path.dirname(__file__), "../conf/model/kbrs.yaml")
            if os.path.exists(yaml_path):
                kbrs_cfg = OmegaConf.load(yaml_path)
                kbrs_params = OmegaConf.to_container(kbrs_cfg.kbrs_params, resolve=True) if hasattr(kbrs_cfg, 'kbrs_params') else OmegaConf.to_container(kbrs_cfg, resolve=True)
            else:
                kbrs_params = {}
            # Inference에 맞춰 동적 파라미터 주입
            kbrs_params["window_size"] = args.window_size
            kbrs_params.setdefault("per_window", 9)
        except Exception as e:
            Logger.warning(f"[Inference] KBRS YAML 로드 실패: {e}")
            kbrs_params = {}

    # 2. 아키텍처, 버전, 사이즈 결정 (pipeline.py -> cli.py 를 통해 넘어온 인자 사용)
    model_name_lower = args.model_name.lower()
    if "centernet" in model_name_lower:
        arch_name = "centernet"
    elif "deformable" in model_name_lower or ("detr" in model_name_lower and "rtdetr" not in model_name_lower):
        arch_name = "deformable_detr"
    elif "rtdetr" in model_name_lower:
        arch_name = "rtdetr"
    else:
        arch_name = "maskrcnn"

    rtdetr_version = getattr(args, "rtdetr_version", "v1")
    rtdetr_size = getattr(args, "rtdetr_size", "l")
     
    # 3. Load model ONCE (reuse across replays)
    model = _load_model(
        model_path=model_path,
        device=device,
        in_channels=in_channels,
        window_size=args.window_size,
        architecture=arch_name,           
        num_classes=2,
        use_kbrs=use_kbrs_flag,         
        kbrs_params=kbrs_params,
        rtdetr_version=rtdetr_version,
        rtdetr_size=rtdetr_size
    )

    input_root = os.path.join(args.data_root, "input", "dst")
    for replay_id in args.replays:
        Logger.info(f"--- Processing replay: {replay_id} ---")

        # Build dataset for this replay
        dataset = InferenceDataset(
            input_root,
            [replay_id],
            window_size=args.window_size,
            include_components=args.include_components
        )

        if 0.0 < args.sample_ratio < 1.0:
            total_len = len(dataset)
            sample_size = int(total_len * args.sample_ratio)
            indices = torch.randperm(total_len).tolist()[:sample_size]
            dataset = Subset(dataset, indices)
            Logger.info(f"[Inference] Applied sampling: {sample_size}/{total_len} frames for replay {replay_id}")

        # Dataloader setup (conservative to avoid RAM issues)
        if args.workers is not None and args.workers >= 0:
            num_workers = args.workers
        else:
            num_workers = _auto_num_workers(device)

        Logger.info(f"[Inference] Dataset frames for {replay_id}: {len(dataset)}; num_workers={num_workers}")

        dl_kwargs = dict(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
            persistent_workers=False,
        )
        data_loader = DataLoader(dataset, **dl_kwargs)
        
        # Run inference
        replay_results = []
        is_tty = sys.stderr.isatty() or sys.stdout.isatty()
        min_interval = 0.1 if is_tty else 5.0
        with torch.inference_mode():
            for images, metas in tqdm.tqdm(data_loader, desc="Running inference for replay", unit="batch", mininterval=min_interval):
                images = [img.to(device, non_blocking=True) for img in images]
                outputs = model(images)

                for output, (rid, frame_id) in zip(outputs, metas):
                    scores_all = output["scores"].detach().cpu().numpy().tolist()
                    if args.score_threshold is not None:
                        keep_idx = [i for i, s in enumerate(scores_all) if s >= args.score_threshold]
                    else:
                        keep_idx = list(range(len(scores_all)))

                    boxes_all = output["boxes"].detach().cpu().numpy().tolist()
                    labels_all = output["labels"].detach().cpu().numpy().tolist()

                    frame_boxes, frame_scores, frame_labels = [], [], []
                    for idx in keep_idx:
                        frame_boxes.append(boxes_all[idx])
                        frame_scores.append(scores_all[idx])
                        frame_labels.append(labels_all[idx])

                    entry = {
                        "frame_id": frame_id,
                        "boxes": frame_boxes,
                        "scores": frame_scores,
                        "labels": frame_labels,
                    }
                    replay_results.append(entry)

                del outputs, images

        # Save predictions (COCO-style, bbox-only)
        save_predictions_as_coco(
            replay_id=replay_id,
            replay_results=replay_results,
            label_method=args.label_method,
            output_dir=run_dir
        )

        # Cleanup per-replay
        del data_loader, dataset, replay_results
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Notify (best-effort)
        try:
            send_message(f"@work [Inference] Completed {replay_id}. Predictions saved at {run_dir}")
        except Exception as e:
            Logger.error(f"[Inference] Error sending message: {e}")

    # Final cleanup
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        
    Logger.info("[Inference] Complete!")


def main():
    args = parse_inference_args()
    run_inference(args)


if __name__ == "__main__":
    main()