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
    A custom dataset for Mask R-CNN inference that reads only input .npy files.
    - Assumes that frames for each replay_id exist in 'data/input/dst/{replay_id}.rep/*.npy'.
    - __getitem__ returns a tuple: (image_tensor, (replay_id, frame_id)).
    """
    def __init__(self, input_root: str, replay_ids: list, include_components: list = None):
        super().__init__()
        self.input_root = input_root

        if include_components:
            self.channel_indices = sorted(sum([config.COMPONENT_CHANNEL_MAP[c] for c in include_components], []))
        else:
            self.channel_indices = list(range(len(config.Channel)))
        
        value_to_name_map = {member.value: name for name, member in config.Channel.__members__.items()}
        channel_names = [value_to_name_map[i] for i in self.channel_indices]
        Logger.info(f"[InferenceDataset] Using {len(self.channel_indices)} channels: {channel_names}")

        # Create a list of (replay_id, frame_id:int, npy_path:str) tuples.
        self.indexes = []
        for rid in map(str, replay_ids):
            rep_dir = os.path.join(self.input_root, f"{rid}.rep")
            if not os.path.isdir(rep_dir):
                Logger.warn(f"[InferenceDataset] Missing directory: {rep_dir}")
                continue

            # Sort all .npy files numerically.
            npy_files = sorted(
                [f for f in os.listdir(rep_dir) if f.endswith(".npy")],
                key=lambda s: int(os.path.splitext(s)[0])
            )
            for fname in npy_files:
                frame_id = int(os.path.splitext(fname)[0])
                npy_path = os.path.join(rep_dir, fname)
                self.indexes.append((rid, frame_id, npy_path))

        if len(self.indexes) == 0:
            raise RuntimeError(f"No .npy files found for replays {replay_ids}. Aborting.")

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        rid, frame_id, npy_path = self.indexes[idx]
        arr = np.load(npy_path)
        arr = arr[self.channel_indices]
        if arr.ndim != 3:
            raise ValueError(f"Unexpected array shape {arr.shape} at {npy_path}")
        img = torch.from_numpy(arr).float()

        return img, (rid, frame_id)


def collate_fn(batch):
    """
    A collate_fn for the DataLoader to bundle a batch into the format ([images], [metadata]).
    The metadata is a list of (replay_id, frame_id) tuples.
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
    Runs inference on a given model and data loader.
    
    Sets the model to evaluation mode, iterates through all frames in the data_loader,
    collects the predicted results (boxes, scores, labels, masks), and returns them
    as a list of dictionaries.

    Returns:
        A list of dictionaries, where each dictionary represents a frame's predictions.
        [
            {
              "frame_id": int,
              "boxes": [[x1,y1,x2,y2], ...],
              "scores": [s1, s2, ...],
              "labels": [l1, l2, ...],
              "masks": [binary_mask, ...],
            },
            ...
        ]
    """
    model.eval()
    replay_results = []

    with torch.no_grad():
        for images, metas in tqdm.tqdm(data_loader, desc="Running inference for replay", unit="batch"):
            # images: list of tensors [C,H,W], metas: list of (rid, frame_id)
            images = [img.to(device) for img in images]
            outputs = model(images)  # list of dict, length = batch_size

            for output, (rid, frame_id) in zip(outputs, metas):
                # Filter predictions based on the score threshold.
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
                    # Convert mask to a binary format.
                    binary_mask = (masks_all[idx, 0] >= 0.5).astype(np.uint8).tolist()
                    frame_masks.append(binary_mask)

                entry = {
                    "frame_id": frame_id,
                    "boxes":    frame_boxes,
                    "scores":   frame_scores,
                    "labels":   frame_labels,
                    "masks":    frame_masks
                }
                replay_results.append(entry)

    return replay_results


def save_predictions_as_coco(
    replay_id: str,
    replay_results: list,
    output_dir: str
):
    """
    Saves the inference results for a single replay to a COCO-formatted JSON file.
    
    Example output file: output_dir/{replay_id}_predictions.json
    """
    os.makedirs(output_dir, exist_ok=True)

    categories = [{"id": 1, "name": "viewport", "supercategory": "viewport"}]

    coco = {
        "info": {"description": f"Predictions for replay {replay_id}", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories
    }

    ann_id = 1
    for item in replay_results:
        fid = item["frame_id"]
        coco["images"].append({
            "id":   fid,
            "file_name": f"{replay_id}.rep/{fid}.npy",
            "width":  config.ORIGIN_SHAPE[1],
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

    out_path = os.path.join(output_dir, f"{replay_id}_predictions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    Logger.info(f"[Inference] Saved predictions for replay {replay_id} -> {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Mask R-CNN inference on preprocessed StarCraft II replays")

    # Data and I/O
    group_data = parser.add_argument_group("Data and I/O")
    group_data.add_argument("--replays", nargs="+", required=True, help="List of replay IDs to run inference on.")
    group_data.add_argument("--data-root", type=str, default=os.path.join(os.getcwd(), "data"), help="Root directory for data.")
    group_data.add_argument("--output-dir", type=str, default=os.path.join(os.getcwd(), "predictions"), help="Directory to save prediction JSON files.")
    group_data.add_argument("--include-components", type=str, nargs='+', default=['worker', 'ground', 'air', 'building', 'vision'], help="List of components to include.")

    # Model Loading
    group_model = parser.add_argument_group("Model Loading")
    group_model.add_argument("--model-root", type=str, default=os.path.join(os.getcwd(), "models"), help="Root directory for model checkpoints.")
    group_model.add_argument("--model-name", type=str, required=True, help="Name of the model folder to use.")
    group_model.add_argument("--model-number", type=int, required=True, help="Checkpoint number to use (e.g., 4 for model_4.pth).")
    group_model.add_argument("--label-method", type=str, default=config.LABEL_METHODS[0], choices=config.LABEL_METHODS, help="Label method for reference (not used in inference).")

    # Inference Hyperparameters
    group_hyper = parser.add_argument_group("Inference Hyperparameters")
    group_hyper.add_argument("--batch-size", type=int, default=8, help="Batch size for inference.")
    group_hyper.add_argument("--score-thr", type=float, default=0.5, help="Objectness score threshold for filtering predictions.")
    group_hyper.add_argument("--sample-ratio", type=float, default=1.0, help="Fraction of frames to sample for inference (0.0 < ratio <= 1.0).")

    return parser.parse_args()


def main():
    args = parse_args()
    Logger.info("[Inference] Starting...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Logger.info(f"[Inference] Using device: {device}")

    # Create model load path
    model_folder = os.path.join(args.model_root, args.model_name)
    model_path = os.path.join(model_folder, f"model_{args.model_number}.pth")
    Logger.info(f"[Inference] Loading model from: {model_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # Load the model (once)
    num_classes = 2  # background + viewport
    
    # Create a temporary dataset to determine the number of channels
    # TODO: This can be improved for efficiency (e.g., by saving metadata with the model)
    temp_input_root = os.path.join(args.data_root, "input", "dst")
    temp_dataset = InferenceDataset(temp_input_root, [args.replays[0]], include_components=args.include_components)
    in_channels = len(temp_dataset.channel_indices)
    
    model = get_model_instance_segmentation(num_classes, in_channels=in_channels, window_size=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Sequentially run inference and save results for each replay
    input_root = os.path.join(args.data_root, "input", "dst")
    for replay_id in args.replays:
        Logger.info(f"--- Processing replay: {replay_id} ---")
        
        # Create dataset and dataloader
        dataset = InferenceDataset(input_root, [replay_id], include_components=args.include_components)
        
        if 0.0 < args.sample_ratio < 1.0:
            total_len = len(dataset)
            sample_size = int(total_len * args.sample_ratio)
            indices = torch.randperm(total_len).tolist()[:sample_size]
            dataset = Subset(dataset, indices)
            Logger.info(f"[Inference] Applied sampling: {sample_size}/{len(dataset.dataset)} frames for replay {replay_id}")

        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )

        # Run inference
        replay_results = run_inference(
            model=model,
            data_loader=data_loader,
            device=device,
            score_threshold=args.score_thr
        )

        # Save results to JSON
        save_predictions_as_coco(
            replay_id=replay_id,
            replay_results=replay_results,
            output_dir=args.output_dir
        )

    Logger.info("[Inference] Complete!")


if __name__ == "__main__":
    main()


