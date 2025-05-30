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
                 window_size: int = 1,
                 indices: list = None,
                 verbose: bool = True):
        self.verbose = verbose
        self.files = []            # (image_id, npy_path) 쌍
        self.window_size = window_size

        for rid in map(str, training_ids):
            inp_dir = os.path.join(input_root, f"{rid}.rep")
            json_file = os.path.join(label_root, f"{rid}.rep", f"{label_method}.json")
            if not os.path.isdir(inp_dir):
                if self.verbose: Logger.warn(f"Missing input dir: {inp_dir}")
                continue
            if not os.path.isfile(json_file):
                if self.verbose: Logger.warn(f"Missing COCO JSON: {json_file}")
                continue

            # JSON 한 번만 로드
            coco = json.load(open(json_file, 'r', encoding='utf-8'))
            # images 리스트 순회하며 (image_id, npy_path) 쌓기
            for img in coco["images"]:
                img_id   = int(img["id"])
                # file_name에 이미 상대경로를 넣었다면 그대로, 아니라면 조합
                npy_path = os.path.join(input_root, f"{rid}.rep", f"{img_id}.npy")
                self.files.append((rid, img_id, npy_path, coco))

        if not self.files:
            raise RuntimeError("Empty dataset")

        # 인덱싱
        if indices is not None:
            self.files = [self.files[i] for i in indices]

        if self.verbose:
            Logger.info(f"Total samples: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rid, img_id, npy_path, coco = self.files[idx]
        
        # 1) 입력 로드
        arr = np.load(npy_path)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected shape {arr.shape}")
        input_tensor = torch.from_numpy(arr).float()  # (9, H, W)

        # 2) 해당 image_id 어노테이션만 필터링
        anns = [a for a in coco["annotations"] if int(a["image_id"]) == img_id]

        boxes, masks, labels, areas, iscrowd = [], [], [], [], []
        H = int(next(img for img in coco["images"] if int(img["id"]) == img_id)["height"])
        W = int(next(img for img in coco["images"] if int(img["id"]) == img_id)["width"])

        for ann in anns:
            x, y, w, h = map(int, ann["bbox"])
            x1, y1, x2, y2 = x, y, x+w, y+h

            # mask 생성 (바운딩박스로 단순화)
            m = np.zeros((H, W), dtype=np.uint8)
            m[y1:y2, x1:x2] = 1

            boxes.append([x1, y1, x2, y2])
            masks.append(torch.from_numpy(m))
            labels.append(int(ann["category_id"]))
            areas.append((x2-x1)*(y2-y1))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        # 3) 텐서 변환
        if boxes:
            boxes   = torch.tensor(boxes, dtype=torch.float32)
            masks   = torch.stack(masks)                 # (N, H, W)
            labels  = torch.tensor(labels, dtype=torch.int64)
            area    = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
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
            "image_id": idx,     # Python int
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