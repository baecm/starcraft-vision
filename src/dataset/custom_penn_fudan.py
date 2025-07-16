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
    def __init__(self, input_root: str, label_root: str, label_method: str, training_ids: list, window_size: int = 1, indices: list = None, training: bool = True, verbose: bool = True):
        self.training = training
        self.verbose = verbose
        self.files = []  # (rid, image_id, npy_path, image_dict, ann_dict)
        self.window_size = window_size

        for rid in map(str, training_ids):
            input_dir = os.path.join(input_root, f"{rid}.rep")
            pkl_file = os.path.join(label_root, f"{rid}.rep", f"{label_method}.pkl")
            if not os.path.isdir(input_dir):
                if self.verbose:
                    Logger.warn(f"Missing input dir: {input_dir}")
                continue
            if not os.path.isfile(pkl_file):
                if self.verbose:
                    Logger.warn(f"Missing Pickle: {pkl_file}")
                continue

            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            image_list = data.get("images", [])
            ann_list = data.get("annotations", [])

            # image_id → image dict
            image_dict = {int(img["id"]): img for img in image_list}
            # image_id → list of annotations
            ann_dict = {}
            for ann in ann_list:
                image_id = int(ann["image_id"])
                ann_dict.setdefault(image_id, []).append(ann)

            for image_id, img_info in image_dict.items():
                npy_path = os.path.join(input_root, f"{rid}.rep", f"{image_id}.npy")
                self.files.append((rid, image_id, npy_path, image_dict, ann_dict))

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

        if not os.path.isfile(npy_path):
            raise FileNotFoundError(f"[CustomDataset] Missing input file: {npy_path}")

        arr = np.load(npy_path)
        if arr.ndim != 3:
            raise ValueError(f"[CustomDataset] Unexpected input shape: {arr.shape} at {npy_path}")
        if not np.isfinite(arr).all():
            Logger.error(f"[CustomDataset] Non-finite input at {npy_path}")
        if np.abs(arr).max() > 1e5:
            Logger.warn(f"[CustomDataset] Unusually large input values in {npy_path}: max={np.abs(arr).max()}")

        input_tensor = torch.from_numpy(arr).float()  # (9, H, W)
        H, W = arr.shape[1], arr.shape[2]

        if not self.training:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, H, W), dtype=torch.uint8),
                "image_id": torch.tensor([img_id]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }
            return input_tensor, target

        img_info = image_dict.get(img_id)
        if img_info is None:
            raise KeyError(f"[CustomDataset] Image ID {img_id} not found in image_dict")

        anns = ann_dict.get(img_id, [])
        H, W = int(img_info["height"]), int(img_info["width"])

        boxes, masks, labels, areas, iscrowd = [], [], [], [], []
        for ann in anns:
            x, y, w, h = map(int, ann["bbox"])
            if w <= 0 or h <= 0:
                Logger.warn(f"[Dataset] Invalid bbox size: {ann['bbox']} → skipping")
                continue
            x1, y1, x2, y2 = x, y, x + w, y + h
            if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
                Logger.warn(f"[Dataset] Invalid bbox coords: {(x1, y1, x2, y2)} @ {npy_path} → skipping")
                continue

            # clip bbox to image boundary
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(0, min(x2, W))
            y2 = max(0, min(y2, H))

            if x2 <= x1 or y2 <= y1:
                Logger.warn(f"[Dataset] Clipped bbox became invalid: {(x1, y1, x2, y2)} → skipping")
                continue

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
            "image_id": torch.tensor([img_id]),
            "area": area,
            "iscrowd": iscrowd
        }

        return input_tensor, target


    def get_coco_structure(self):
        used_image_ids = set(f[1] for f in self.files)  # self.files에서 실제 사용하는 image_id만

        images = []
        annotations = []
        categories = []
        
        sample_file = self.files[0]
        _, _, _, image_dict, ann_dict = sample_file

        for idx, (rid, img_id, npy_path, image_dict, ann_dict) in enumerate(self.files):
            img_info = image_dict[img_id].copy()
            img_info["id"] = img_id  # 강제 재정의 (GT 기준)
            images.append(img_info)

            anns = ann_dict.get(img_id, [])
            for ann in anns:
                ann_copy = ann.copy()
                ann_copy["image_id"] = img_id  # 강제 재정의 (GT 기준)
                annotations.append(ann_copy)

                cid = ann.get("category_id", 1)
                if not any(c["id"] == cid for c in categories):
                    categories.append({"id": cid, "name": f"class_{cid}"})

        return {
            "info": {"description": "autogen", "version": "1.0"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
            
    def preprocessing(self, *data):
        if not isinstance(data[0], np.ndarray) or data[0].ndim != 3:
            Logger.error(f"[CustomDataset] Invalid frame data: shape={getattr(data[0], 'shape', None)}")
            raise ValueError("Frame data must be 3D NumPy array")
        temp = np.zeros([self.window_size, 9, data[0].shape[1], data[0].shape[2]])
        for i, d in enumerate(data):
            temp[i] = d
        data = temp.reshape(self.window_size * temp.shape[1], temp.shape[2], -1)
        return torch.FloatTensor(data)
