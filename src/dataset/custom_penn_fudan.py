import os
import re
import glob
import json
import numpy as np
import torch
from .penn_fudan import PennFudanDataset as BasePennFudanDataset
from utils.logger import Logger


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r"(\d+)", os.path.basename(s))]


class CustomPennFudanDataset(BasePennFudanDataset):
    def __init__(self,
                 input_root: str,
                 label_root: str,
                 label_method: str,
                 training_ids: list,
                 window_size: int = None,
                 indices: list = None,
                 verbose: bool = True):
        """
        - input_root: "data/input/dst"
        - label_root: "data/label/dst"
        - label_method: folder under label_root/{id}.rep/{method}
        - training_ids: list of replay IDs
        - verbose: control logging
        """
        self.verbose = verbose
        if self.verbose:
            Logger.info("[CustomDataset] Initializing from input and JSON label directories...")
        self.files = []
        self.window_size = window_size or 1

        for replay_id in map(str, training_ids):
            inp_dir = os.path.join(input_root, f"{replay_id}.rep")
            ann_dir = os.path.join(label_root, f"{replay_id}.rep", label_method)
            if not os.path.isdir(inp_dir):
                if self.verbose:
                    Logger.warn(f"[CustomDataset] Missing input directory: {inp_dir}")
                continue
            if not os.path.isdir(ann_dir):
                if self.verbose:
                    Logger.warn(f"[CustomDataset] Missing annotation directory: {ann_dir}")
                continue

            npy_files = sorted(glob.glob(os.path.join(inp_dir, "*.npy")),
                               key=natural_sort_key)
            if self.verbose:
                Logger.info(f"[CustomDataset] {replay_id}: found {len(npy_files)} input files")

            for npy_path in npy_files:
                base = os.path.splitext(os.path.basename(npy_path))[0]
                json_path = os.path.join(ann_dir, f"{base}.json")
                if os.path.isfile(json_path):
                    self.files.append((npy_path, json_path))
                elif self.verbose:
                    Logger.warn(f"[CustomDataset] Missing JSON for {npy_path}")

        if not self.files:
            if self.verbose:
                Logger.error("[CustomDataset] No valid samples found. Aborting.")
            raise RuntimeError("Empty dataset")

        # apply optional indexing
        if indices is not None:
            self.files = [self.files[i] for i in indices]
            self.indices = indices
        else:
            self.indices = list(range(len(self.files)))

        if self.verbose:
            Logger.info(f"[CustomDataset] Total samples: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npy_path, json_path = self.files[idx]

        # 1) load the raw input array
        arr = np.load(npy_path)   # arr.shape == (9, 128, 128)

        # 2) ensure shape is (C, H, W)
        if arr.ndim == 3:
            input_tensor = torch.from_numpy(arr).float()
        else:
            raise ValueError(f"Unexpected npy shape: {arr.shape}")

        # 이후 JSON 로드 및 target 구성
        anno = json.load(open(json_path, 'r', encoding='utf-8'))
        image_info = anno['images'][0]
        W, H = image_info['width'], image_info['height']
        anns = anno.get('annotations', [])

        masks, boxes, labels = [], [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            mask = np.zeros((H, W), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1
            masks.append(mask)
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])

        # 빈 케이스 처리
        if masks:
            boxes   = torch.tensor(boxes, dtype=torch.float32)
            masks   = torch.stack([torch.from_numpy(m) for m in masks])
            labels  = torch.tensor(labels, dtype=torch.int64)
            area    = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            boxes   = torch.zeros((0,4), dtype=torch.float32)
            masks   = torch.zeros((0, H, W), dtype=torch.uint8)
            labels  = torch.zeros((0,), dtype=torch.int64)
            area    = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "masks":    masks,
            "image_id": torch.tensor([idx]),
            "area":     area,
            "iscrowd":  iscrowd
        }
        return input_tensor, target

    def preprocessing(self, *data):
        if not isinstance(data[0], np.ndarray) or data[0].ndim != 3:
            Logger.error(f"[CustomDataset] Invalid frame data: shape={getattr(data[0], 'shape', None)}")
            raise ValueError("Frame data must be 3D NumPy array")
        temp = np.zeros([self.window_size, 9, data[0].shape[1], data[0].shape[2]])
        for i, d in enumerate(data):
            temp[i] = d
        data = temp.reshape(self.window_size * temp.shape[1], temp.shape[2], -1)
        return torch.FloatTensor(data)