import os
import re
import glob
import numpy as np
import torch
from .penn_fudan import PennFudanDataset as BasePennFudanDataset
from utils.logger import Logger


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", os.path.basename(s))]
    

class CustomPennFudanDataset(BasePennFudanDataset):
    def __init__(self, root_dir: str, label_method: str, training: list, window_size: int = None):
        Logger.info("[CustomDataset] Initializing from preprocessed pair directory...")

        self.files = []
        for replay_id in map(str, training):
            pair_dir = os.path.join(root_dir, f"{replay_id}", label_method)
            if not os.path.isdir(pair_dir):
                Logger.warn(f"[CustomDataset] Skipping {replay_id} (no path: {pair_dir})")
                continue

            npy_files = sorted(glob.glob(os.path.join(pair_dir, "*.npy")), key=natural_sort_key)
            Logger.info(f"[CustomDataset] {replay_id}: found {len(npy_files)} pair files")
            self.files.extend(npy_files)

        if not self.files:
            Logger.error("[CustomDataset] No pair data found. Aborting.")
            raise RuntimeError("Empty dataset")

        Logger.info(f"[CustomDataset] Total samples: {len(self.files)}")

    def _generate_sequence_indices(self):
        seq_indexs = []
        index_a = 0

        for dir_path in self.dir_paths:
            all_files = os.listdir(dir_path)
            num_files = len(all_files)
            valid_length = num_files - 150  # usable frame 수

            Logger.debug(f"[CustomDataset] dir: {dir_path}, total files: {num_files}, valid: {valid_length}")

            if valid_length <= 0:
                Logger.warn(f"[CustomDataset] Skipping dir: {dir_path} (not enough usable frames)")
                continue

            index_b = index_a + valid_length
            seq_indexs.append((dir_path, index_a, index_b))
            index_a = index_b

        Logger.debug(f"[CustomDataset] Final seq_indexs: {seq_indexs}")
        return seq_indexs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            pair = np.load(path, allow_pickle=True)
            input_arr, mask_arr = pair[0], pair[1]
        except Exception as e:
            Logger.error(f"[CustomDataset] Failed to load pair at {path}: {e}")
            raise

        # [1] Input 처리
        if isinstance(input_arr, list):
            input_arr = np.stack(input_arr)

        if input_arr.ndim == 4:
            # (T, C, H, W) → (C*T, H, W)
            input_arr = input_arr.transpose(1, 0, 2, 3).reshape(-1, input_arr.shape[2], input_arr.shape[3])
        elif input_arr.ndim == 3:
            # (T, H, W) → (1*T, H, W)
            input_arr = input_arr.reshape(-1, input_arr.shape[1], input_arr.shape[2])

        input_tensor = torch.FloatTensor(input_arr)

        # [2] Masks와 Boxes
        masks = np.array(mask_arr)
        num_objs = len(masks)
        boxes = []

        for i in range(num_objs):
            pos = np.where(masks[i])
            if pos[0].size == 0 or pos[1].size == 0:
                boxes.append([0, 0, 1, 1])  # fallback
            else:
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        return input_tensor, target

    def _load_data(self, dir_path, real_idx):
        Logger.debug(f"[CustomDataset] Loading frames from {dir_path} at index {real_idx}")
        try:
            frames = [np.load(os.path.join(dir_path, f"{real_idx + i}.npy"), allow_pickle=True)[0] for i in range(self.window_size)]
            masks = np.load(os.path.join(dir_path, f"{real_idx + 150}.npy"), allow_pickle=True)[1]
            Logger.debug(f"[CustomDataset] Loaded frame shapes: {[f.shape if hasattr(f, 'shape') else type(f) for f in frames]}")
        except Exception as e:
            Logger.error(f"[CustomDataset] Error loading data: {e}")
            raise

        input_data = self.preprocessing(*frames)
        boxes = self._extract_boxes(masks)
        target = self._create_target(boxes, masks, real_idx)

        return input_data, target

    def _extract_boxes(self, masks):
        boxes = []
        for i, mask in enumerate(masks):
            pos = np.where(mask)
            if pos[0].size == 0 or pos[1].size == 0:
                Logger.warn(f"[CustomDataset] Empty mask at index {i}")
                boxes.append([0, 0, 1, 1])  # fallback
                continue
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        return torch.tensor(boxes, dtype=torch.float32)

    def _create_target(self, boxes, masks, idx):
        num_objs = len(boxes)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        Logger.debug(f"[CustomDataset] Creating target for idx {idx} with {num_objs} objects")

        target = {
            "boxes": boxes,
            "labels": torch.ones((num_objs,), dtype=torch.int64),
            "masks": torch.tensor(masks, dtype=torch.uint8),
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((num_objs,), dtype=torch.int64)
        }

        return target

    def preprocessing(self, *data):
        if not isinstance(data[0], np.ndarray) or len(data[0].shape) != 3:
            Logger.error(f"[CustomDataset] Invalid input frame: type={type(data[0])}, shape={getattr(data[0], 'shape', None)}")
            raise ValueError("Frame data must be 3D NumPy arrays")

        temp = np.zeros([self.window_size, 9, data[0].shape[1], data[0].shape[2]])
        for i, d in enumerate(data):
            temp[i] = d

        data = temp.reshape(self.window_size * temp.shape[1], temp.shape[2], -1)
        return torch.FloatTensor(data)
