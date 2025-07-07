import os
import re
import pickle
import numpy as np
import torch
from .penn_fudan import PennFudanDataset as BasePennFudanDataset
from utils.logger import Logger


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", os.path.basename(s))]


class CustomPennFudanDataset(BasePennFudanDataset):
    def __init__(self, input_root: str, label_root: str, label_method: str, training_ids: list, window_size: int = 1, indices: list = None, verbose: bool = True):
        self.verbose = verbose
        self.files = []  # (rid, image_id, npy_path, image_dict, ann_dict)
        self.window_size = window_size

        for rid in map(str, training_ids):
            input_dir = os.path.join(input_root, f"{rid}.rep")
            pkl_file = os.path.join(label_root, f"{rid}.rep", f"{label_method}.pkl")
            if not os.path.isdir(input_dir):
                if self.verbose: Logger.warn(f"Missing input dir: {input_dir}")
                continue
            if not os.path.isfile(pkl_file):
                if self.verbose: Logger.warn(f"Missing Pickle: {pkl_file}")
                continue

            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
                image_dict = data["images"]
                ann_dict = data["annotations"]

            for img_id in image_dict:
                npy_path = os.path.join(input_root, f"{rid}.rep", f"{img_id}.npy")
                self.files.append((rid, img_id, npy_path, image_dict, ann_dict))

        if not self.files:
            raise RuntimeError("Empty dataset")

        if indices is not None:
            self.files = [self.files[i] for i in indices]

        if self.verbose:
            Logger.info(f"Total samples: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rid, img_id, npy_path, image_dict, ann_dict = self.files[idx]

        arr = np.load(npy_path)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected shape {arr.shape}")
        input_tensor = torch.from_numpy(arr).float()  # (9, H, W)

        anns = ann_dict.get(img_id, [])
        img_info = image_dict[img_id]
        H, W = int(img_info["height"]), int(img_info["width"])

        boxes, masks, labels, areas, iscrowd = [], [], [], [], []

        for ann in anns:
            x, y, w, h = map(int, ann["bbox"])
            x1, y1, x2, y2 = x, y, x + w, y + h

            m = np.zeros((H, W), dtype=np.uint8)
            m[y1:y2, x1:x2] = 1

            boxes.append([x1, y1, x2, y2])
            masks.append(torch.from_numpy(m))
            labels.append(int(ann["category_id"]))
            areas.append((x2 - x1) * (y2 - y1))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            masks = torch.stack(masks)
            labels = torch.tensor(labels, dtype=torch.int64)
            area = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, H, W), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

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
        if not isinstance(data[0], np.ndarray) or data[0].ndim != 3:
            Logger.error(f"[CustomDataset] Invalid frame data: shape={getattr(data[0], 'shape', None)}")
            raise ValueError("Frame data must be 3D NumPy array")
        temp = np.zeros([self.window_size, 9, data[0].shape[1], data[0].shape[2]])
        for i, d in enumerate(data):
            temp[i] = d
        data = temp.reshape(self.window_size * temp.shape[1], temp.shape[2], -1)
        return torch.FloatTensor(data)
