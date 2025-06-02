#!/usr/bin/env python
# src/inference.py

import os
import argparse
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import tqdm

from model.maskrcnn_builder import get_model_instance_segmentation
import detection.transforms as T
from utils.logger import Logger
import config


class InferenceDataset(Dataset):
    """
    순수 입력(npy)만 읽어서 Mask R-CNN 추론을 수행할 수 있도록 한 커스텀 데이터셋입니다.
    - 각 replay_id별로 'data/input/dst/{replay_id}.rep/*.npy' 형태로 프레임이 존재한다고 가정합니다.
    - __getitem__은 (image_tensor, (replay_id, frame_id)) 튜플을 반환합니다.
    """
    def __init__(self, input_root: str, replay_ids: list):
        super().__init__()
        self.input_root = input_root

        # [(replay_id, frame_id:int, npy_path:str), ...] 리스트를 만듭니다.
        self.indexes = []
        for rid in map(str, replay_ids):
            rep_dir = os.path.join(self.input_root, f"{rid}.rep")
            if not os.path.isdir(rep_dir):
                Logger.warn(f"[InferenceDataset] Missing directory: {rep_dir}")
                continue

            # 모든 .npy 파일을 숫자 순으로 정렬
            npy_files = sorted(
                [f for f in os.listdir(rep_dir) if f.endswith(".npy")],
                key=lambda s: int(os.path.splitext(s)[0])
            )
            for fname in npy_files:
                frame_id = int(os.path.splitext(fname)[0])
                npy_path = os.path.join(rep_dir, fname)
                self.indexes.append((rid, frame_id, npy_path))

        if len(self.indexes) == 0:
            raise RuntimeError("No .npy files found for inference. Aborting.")

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        rid, frame_id, npy_path = self.indexes[idx]
        arr = np.load(npy_path)            # shape = (9, H, W)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected array shape {arr.shape} at {npy_path}")
        img = torch.from_numpy(arr).float()  # (9, H, W)

        return img, (rid, frame_id)


def collate_fn(batch):
    """
    DataLoader collate_fn: batch 의 형식을 ([images], [metadata]) 로 묶어 주기 위함.
    metadata는 list of (rid, frame_id) 튜플.
    """
    images, metas = zip(*batch)
    return list(images), list(metas)


def run_inference(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.5
):
    """
    모델을 평가 모드로 두고, data_loader 안의 모든 프레임을 순회하면서
    예측된 결과(박스, 스코어, 클래스, 마스크)를 수집하여, replay_id별로 리턴합니다.

    리턴값: {
       replay_id_1: [
         {
           "frame_id": int,
           "boxes": [[x1,y1,x2,y2], ...],
           "scores": [s1, s2, ...],
           "labels": [l1, l2, ...],
           "masks": [mask_binary, ...],
         },
         ...
       ],
       replay_id_2: [ ... ],
       ...
    }
    """
    model.eval()
    results = {}  # replay_id → list of frame‐level 딕셔너리들

    with torch.no_grad():
        for images, metas in tqdm.tqdm(data_loader, desc="Running inference", unit="batch"):
            # images: list of tensors [C,H,W], metas: list of (rid, frame_id)
            images = [img.to(device) for img in images]
            outputs = model(images)  # list of dict, length = batch_size

            for output, (rid, frame_id) in zip(outputs, metas):
                # 예측 필터링: score >= threshold
                scores_all = output["scores"].cpu().numpy().tolist()
                keep_idx = [i for i, s in enumerate(scores_all) if s >= score_threshold]

                boxes_all  = output["boxes"].cpu().numpy().tolist()
                labels_all = output["labels"].cpu().numpy().tolist()
                masks_all  = output["masks"].cpu().numpy()  # (N, 1, H, W)

                frame_boxes  = []
                frame_scores = []
                frame_labels = []
                frame_masks  = []

                for idx in keep_idx:
                    frame_boxes.append(boxes_all[idx])
                    frame_scores.append(scores_all[idx])
                    frame_labels.append(labels_all[idx])
                    # mask를 바이너리로 변환
                    binary_mask = (masks_all[idx, 0] >= 0.5).astype(np.uint8).tolist()
                    frame_masks.append(binary_mask)

                entry = {
                    "frame_id": frame_id,
                    "boxes":    frame_boxes,   # [[x1,y1,x2,y2], ...]
                    "scores":   frame_scores,  # [float, ...]
                    "labels":   frame_labels,  # [int, ...]
                    "masks":    frame_masks    # list of (H×W) 0/1 이중 리스트
                }

                if rid not in results:
                    results[rid] = []
                results[rid].append(entry)

    return results


def save_predictions_as_coco(
    all_results: dict,
    output_dir: str
):
    """
    all_results 형식:
    {
      replay_id1: [ 
         {"frame_id": int, "boxes": [...], "scores": [...], "labels": [...], "masks": [...]}, 
         … 
      ],
      replay_id2: [ ... ],
      ...
    }

    이를 COCO evaluation과 호환되는 JSON으로 출력합니다.
    - 각 replay_id별로 단일 JSON을 생성하며,
      images, annotations, categories 필드만 채웁니다.
    - 예시 output: output_dir/{replay_id}_predictions.json
    """
    os.makedirs(output_dir, exist_ok=True)

    categories = [{"id": 1, "name": "viewport", "supercategory": "viewport"}]

    for rid, frames in tqdm.tqdm(all_results.items(), desc="Saving predictions", unit="replay"):
        coco = {
            "info": {"description": f"Predictions for replay {rid}", "version": "1.0"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories
        }

        ann_id = 1
        for item in frames:
            fid = item["frame_id"]
            coco["images"].append({
                "id":   fid,
                "file_name": f"{rid}.rep/{fid}.npy",
                "width":  config.ORIGIN_SHAPE[1],  # ORIGIN_SHAPE = (H, W)
                "height": config.ORIGIN_SHAPE[0]
            })
            for box, score, label, _mask in zip(
                item["boxes"], item["scores"], item["labels"], item["masks"]
            ):
                x1, y1, x2, y2 = map(int, box)
                w = x2 - x1
                h = y2 - y1
                segmentation = [[x1, y1, x1 + w, y1, x1 + w, y1 + h, x1, y1 + h]]
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": fid,
                    "category_id": int(label),
                    "bbox": [x1, y1, w, h],
                    "score": float(score),
                    "area": w * h,
                    "segmentation": segmentation,
                    "iscrowd": 0
                })
                ann_id += 1

        out_path = os.path.join(output_dir, f"{rid}_predictions.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)

        Logger.info(f"[Inference] Saved predictions for replay {rid} → {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Mask R-CNN inference on preprocessed StarCraft II replays"
    )
    parser.add_argument(
        "--replays", nargs="+", required=True,
        help="List of replay IDs to run inference on"
    )
    parser.add_argument(
        "--model-root", type=str, default=os.path.join(os.getcwd(), "models"),
        help="모델 체크포인트가 저장된 최상위 디렉토리 (default=models)"
    )
    parser.add_argument(
        "--model-name", type=str, required=True,
        help="사용할 모델 폴더 이름"
    )
    parser.add_argument(
        "--model-number", type=int, required=True,
        help="몇 번째 체크포인트를 사용할지 (예: 4 → model_4.pth)"
    )
    parser.add_argument(
        "--label-method", type=str, default=config.LABEL_METHODS[0],
        choices=config.LABEL_METHODS,
        help="(참고용) GT 레이블 메소드. Inference 에선 사용하지 않지만, 출력 파일 네임에 포함해두면 편합니다."
    )
    parser.add_argument(
        "--data-root", type=str, default=os.path.join(os.getcwd(), "data"),
        help="Data root 위치 (default=data)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Inference 시 배치 크기"
    )
    parser.add_argument(
        "--score-thr", type=float, default=0.5,
        help="Objectness 스코어 임계값"
    )
    parser.add_argument(
        "--sample-ratio", type=float, default=1.0,
        help="샘플링 비율 (0.0 < 샘플링 비율 ≤ 1.0). 1.0이면 전체 프레임 사용."
    )
    parser.add_argument(
        "--output-dir", type=str, default=os.path.join(os.getcwd(), "predictions"),
        help="출력 JSON을 저장할 디렉토리"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    Logger.info("[Inference] Starting...")

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Logger.info(f"[Inference] Using device: {device}")

    # 데이터셋 준비
    input_root = os.path.join(args.data_root, "input", "dst")
    dataset = InferenceDataset(input_root, args.replays)

    # sample_ratio이 1.0 미만인 경우 랜덤 샘플링
    if 0.0 < args.sample_ratio < 1.0:
        total_len = len(dataset)
        sample_size = int(total_len * args.sample_ratio)
        indices = torch.randperm(total_len).tolist()[:sample_size]
        dataset = Subset(dataset, indices)
        Logger.info(f"[Inference] Applied sampling: {sample_size}/{total_len} frames")

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # 모델 로드 경로 생성
    model_folder = os.path.join(args.model_root, args.model_name)
    model_path = os.path.join(model_folder, f"model_{args.model_number}.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # 모델 불러오기
    num_classes = 2  # background + viewport
    model = get_model_instance_segmentation(num_classes, window_size=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 추론 수행
    Logger.info("[Inference] Running inference …")
    all_results = run_inference(
        model=model,
        data_loader=data_loader,
        device=device,
        score_threshold=args.score_thr
    )

    # 결과 JSON 저장
    save_predictions_as_coco(
        all_results=all_results,
        output_dir=args.output_dir
    )

    Logger.info("[Inference] Complete!")


if __name__ == "__main__":
    main()
