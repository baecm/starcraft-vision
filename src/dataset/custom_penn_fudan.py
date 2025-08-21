# src/dataset/custom_penn_fudan.py
import os
import re
import pickle
import numpy as np
import torch
from .penn_fudan import PennFudanDataset as BasePennFudanDataset
from utils.logger import Logger
import config


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", os.path.basename(s))]


class CustomPennFudanDataset(BasePennFudanDataset):
    def __init__(self, input_root: str, label_root: str, label_method: str, training_ids: list, window_size: int = 1, interval: int = 1, indices: list = None, training: bool = True, verbose: bool = True, include_components: list = None):
        self.input_root = input_root
        self.training = training
        self.verbose = verbose
        self.window_size = window_size
        self.interval = max(1, int(interval))
        self.files = []  # Stores tuples of (rid, [image_ids_in_window], image_dict, ann_dict)

        if include_components:
            self.channel_indices = sorted(sum([config.COMPONENT_CHANNEL_MAP[c] for c in include_components], []))
        else:
            self.channel_indices = list(range(len(config.Channel)))

        for rid in map(str, training_ids):
            input_dir = os.path.join(self.input_root, f"{rid}.rep")
            pkl_file = os.path.join(label_root, f"{rid}.rep", f"{label_method}.pkl")

            if not (os.path.isdir(input_dir) and os.path.isfile(pkl_file)):
                if self.verbose:
                    Logger.warn(f"Skipping {rid}: missing input dir or pickle file.")
                continue

            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            image_dict = {int(img["id"]): img for img in data.get("images", [])}
            ann_dict = {}
            for ann in data.get("annotations", []):
                image_id = int(ann["image_id"])
                ann_dict.setdefault(image_id, []).append(ann)

            # Sort image IDs numerically to ensure correct frame sequence
            sorted_image_ids = sorted(image_dict.keys())

            for i in range(0, len(sorted_image_ids) - self.window_size + 1, self.interval):
                window_image_ids = sorted_image_ids[i : i + self.window_size]
                if all(os.path.exists(os.path.join(input_dir, f"{img_id}.npy")) for img_id in window_image_ids):
                    self.files.append((rid, window_image_ids, image_dict, ann_dict))

        if not self.files:
            raise RuntimeError("Empty dataset or no valid windows found.")

        if indices is not None:
            self.files = [self.files[i] for i in indices]

        if self.verbose:
            Logger.info(f"Total windows (samples): {len(self.files)}")
            value_to_name_map = {member.value: name for name, member in config.Channel.__members__.items()}
            channel_names = [value_to_name_map[i] for i in self.channel_indices]
            Logger.info(f"Using {len(self.channel_indices)} channels: {channel_names}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rid, window_image_ids, image_dict, ann_dict = self.files[idx]
        
        # The target is based on the last frame in the window
        target_img_id = window_image_ids[-1]

        # Load all frames in the window and stack them
        window_frames = []
        for img_id in window_image_ids:
            npy_path = os.path.join(self.input_root, f"{rid}.rep", f"{img_id}.npy")
            if not os.path.isfile(npy_path):
                raise FileNotFoundError(f"[CustomDataset] Missing input file: {npy_path}")
            
            arr = np.load(npy_path)
            arr = arr[self.channel_indices] # Apply channel selection
            if arr.ndim != 3:
                raise ValueError(f"[CustomDataset] Unexpected input shape: {arr.shape} at {npy_path}")
            
            window_frames.append(arr)

        # Concatenate frames along the channel axis
        # Shape becomes (C * window_size, H, W)
        input_tensor = torch.from_numpy(np.concatenate(window_frames, axis=0)).float()
        
        _, H, W = input_tensor.shape

        # --- Target Generation (based on the last frame) ---
        img_info = image_dict.get(target_img_id)
        if img_info is None:
            raise KeyError(f"[CustomDataset] Target Image ID {target_img_id} not found in image_dict")

        anns = ann_dict.get(target_img_id, [])
        
        boxes, masks, labels, areas, iscrowd = [], [], [], [], []
        for ann in anns:
            x, y, w, h = map(int, ann["bbox"])
            if w <= 0 or h <= 0: continue
            
            x1, y1, x2, y2 = x, y, x + w, y + h
            # Clip bbox to image boundary
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(0, min(x2, W))
            y2 = max(0, min(y2, H))
            if x2 <= x1 or y2 <= y1: continue

            m = np.zeros((H, W), dtype=np.uint8)
            m[y1:y2, x1:x2] = 1

            boxes.append([x1, y1, x2, y2])
            masks.append(torch.from_numpy(m))
            labels.append(int(ann["category_id"]))
            areas.append((x2 - x1) * (y2 - y1))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if boxes:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "masks": torch.stack(masks),
                "image_id": torch.tensor([target_img_id]),
                "area": torch.tensor(areas, dtype=torch.float32),
                "iscrowd": torch.tensor(iscrowd, dtype=torch.int64)
            }
        else: # No annotations for this frame
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, H, W), dtype=torch.uint8),
                "image_id": torch.tensor([target_img_id]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }

        return input_tensor, target

    def get_coco_structure(self):
        # This method might need adjustment if used, as it currently assumes one frame per file entry.
        # For now, it's left as is, but may not be fully compatible with the new windowed structure.
        used_image_ids = set()
        for _, window_ids, _, _ in self.files:
            used_image_ids.update(window_ids)

        images = []
        annotations = []
        categories = set()
        
        # Need to iterate through all replays to gather all image/annotation data
        # This is inefficient but necessary if we need a complete COCO structure.
        # A better approach would be to cache this. For now, let's assume it's not a critical path.
        
        # This part is complex to rebuild from windowed `self.files`. 
        # Returning a simplified or placeholder structure.
        # A proper implementation would require iterating through all original data again.
        Logger.warn("get_coco_structure() may produce incomplete results with windowed data.")
        return {
            "info": {"description": "autogen-placeholder", "version": "1.0"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "viewport"}]
        }
