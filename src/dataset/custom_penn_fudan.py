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
    def __init__(self, root_dir: str, label_method: str, training: list, window_size: int = None, indices: list = None):
        Logger.info("[CustomDataset] Initializing from preprocessed pair directory...")

        self.files = []
        self.window_size = window_size if window_size is not None else 1

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

        if indices is not None:
            self.files = [self.files[i] for i in indices]
            self.indices = indices
        else:
            self.indices = list(range(len(self.files)))

        Logger.info(f"[CustomDataset] Total samples: {len(self.files)}")

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

        # (1) 입력 처리
        if isinstance(input_arr, list):
            input_arr = np.stack(input_arr)

        if input_arr.ndim == 4:
            # shape: (T, C, H, W) -> (C*T, H, W)
            input_arr = input_arr.transpose(1, 0, 2, 3).reshape(-1, input_arr.shape[2], input_arr.shape[3])
        elif input_arr.ndim == 3:
            input_arr = input_arr.reshape(-1, input_arr.shape[1], input_arr.shape[2])

        input_tensor = torch.FloatTensor(input_arr)

        # (2) 마스크와 바운딩 박스 처리
        masks = np.array(mask_arr)
        num_objs = len(masks)
        boxes = []

        for i in range(num_objs):
            pos = np.where(masks[i])
            if pos[0].size == 0 or pos[1].size == 0:
                boxes.append([0, 0, 1, 1])
            else:
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.tensor(masks, dtype=torch.uint8)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # (3) image_id는 반드시 int 형으로 (COCOEvaluator 호환)
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": idx,
            "area": area,
            "iscrowd": iscrowd
        }

        return input_tensor, target


    def preprocessing(self, *data):
        if not isinstance(data[0], np.ndarray) or len(data[0].shape) != 3:
            Logger.error(f"[CustomDataset] Invalid input frame: type={type(data[0])}, shape={getattr(data[0], 'shape', None)}")
            raise ValueError("Frame data must be 3D NumPy arrays")

        temp = np.zeros([self.window_size, 9, data[0].shape[1], data[0].shape[2]])
        for i, d in enumerate(data):
            temp[i] = d

        data = temp.reshape(self.window_size * temp.shape[1], temp.shape[2], -1)
        return torch.FloatTensor(data)