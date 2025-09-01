# src/dataset/custom_penn_fudan.py
import os
import re
import pickle
import numpy as np
import torch
from .penn_fudan import PennFudanDataset as BasePennFudanDataset
from utils.logger import Logger
import config


class CustomPennFudanDataset(BasePennFudanDataset):
    """
    Windowed dataset:
      - 입력: <input_root>/<rid>.rep/<image_id>.npy  (shape: [C,H,W])
      - 라벨: <label_root>/<rid>.rep/<label_method>.pkl (COCO fields: images, annotations)
      - window_size, interval 규칙으로 이미지 id 윈도우 생성 후, 채널축으로 concat → (C*W, H, W)
      - 타깃은 윈도우의 마지막 프레임 기준으로 bbox→mask를 생성
    """
    def __init__(
        self,
        input_root: str,
        label_root: str,
        label_method: str,
        training_ids: list,
        window_size: int = 1,
        interval: int = 1,
        indices: list = None,
        training: bool = True,
        verbose: bool = True,
        include_components: list = None,
        trim_tail: int = 0
    ):
        self.input_root = input_root
        self.training = training
        self.verbose = verbose
        self.window_size = int(window_size)
        self.interval = max(1, int(interval))
        self.trim_tail = max(0, int(trim_tail))

        # 채널 인덱스 확정
        self.channel_indices = self._build_channel_indices(include_components)

        # (rid, [window_img_ids], image_dict, ann_dict) 튜플 리스트
        self.files = []

        # 리플레이 단위로 메타 로딩 및 윈도우 구성
        for rid in map(str, training_ids):
            input_dir = os.path.join(self.input_root, f"{rid}.rep")
            pkl_path = os.path.join(label_root, f"{rid}.rep", f"{label_method}.pkl")

            if not (os.path.isdir(input_dir) and os.path.isfile(pkl_path)):
                if self.verbose:
                    Logger.warn(f"Skipping {rid}: missing input dir or pickle file.")
                continue

            image_dict, ann_dict = self._load_meta_from_pickle(pkl_path)

            sorted_image_ids = sorted(image_dict.keys())
            if self.trim_tail > 0 and len(sorted_image_ids) > self.trim_tail:
                sorted_image_ids = sorted_image_ids[:-self.trim_tail]

            # 윈도우 생성
            windows = self._generate_windows(sorted_image_ids, self.window_size, self.interval)
            # 유효 윈도우만 필터(모든 .npy 존재)
            for win in windows:
                if self._check_window_files_exist(input_dir, win):
                    self.files.append((rid, win, image_dict, ann_dict))

        if not self.files:
            raise RuntimeError("Empty dataset or no valid windows found.")

        if indices is not None:
            self.files = [self.files[i] for i in indices]

        if self.verbose:
            Logger.info(f"Total windows (samples): {len(self.files)}")
            value_to_name_map = {member.value: name for name, member in config.Channel.__members__.items()}
            channel_names = [value_to_name_map[i] for i in self.channel_indices]
            Logger.info(f"Using {len(self.channel_indices)} channels: {channel_names}")

    # -----------------------------
    # Public API
    # -----------------------------
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rid, window_image_ids, image_dict, ann_dict = self.files[idx]

        # 입력 텐서 생성: (C * window_size, H, W)
        input_tensor = self._concat_window_frames(
            root=self.input_root,
            rid=rid,
            image_ids=window_image_ids,
            channel_indices=self.channel_indices
        ).float()

        _, H, W = input_tensor.shape

        # 타깃: 윈도우 마지막 프레임
        target_img_id = window_image_ids[-1]
        if target_img_id not in image_dict:
            raise KeyError(f"[CustomDataset] Target Image ID {target_img_id} not found in image_dict")

        anns = ann_dict.get(target_img_id, [])
        target = self._make_target_from_anns(anns, H, W, target_img_id)

        return input_tensor, target

    def get_coco_structure(self):
        """
        참고: 윈도우드 구조에선 완전한 COCO 구조를 재구성하기 어렵다.
        필요 시 외부에서 별도 캐시를 구성하거나 이 메서드를 커스터마이즈 하자.
        """
        Logger.warn("get_coco_structure() may produce incomplete results with windowed data.")
        return {
            "info": {"description": "autogen-placeholder", "version": "1.0"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "viewport"}]
        }

    # -----------------------------
    # Class helpers
    # -----------------------------
    @classmethod
    def from_replay_ids(
        cls,
        input_root: str,
        label_root: str,
        label_method: str,
        replay_ids: list,
        **kwargs
    ):
        """
        대체 생성자: 인자로 받은 replay_ids를 그대로 training_ids로 사용.
        """
        return cls(
            input_root=input_root,
            label_root=label_root,
            label_method=label_method,
            training_ids=replay_ids,
            **kwargs
        )

    # -----------------------------
    # Static / Internal helpers
    # -----------------------------
    @staticmethod
    def _build_channel_indices(include_components):
        if include_components:
            # 구성요소 → 채널 인덱스 리스트
            return sorted(sum([config.COMPONENT_CHANNEL_MAP[c] for c in include_components], []))
        # 전 채널 사용
        return list(range(len(config.Channel)))

    @staticmethod
    def _load_meta_from_pickle(pkl_path: str):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        image_dict = {int(img["id"]): img for img in data.get("images", [])}
        ann_dict = {}
        for ann in data.get("annotations", []):
            image_id = int(ann["image_id"])
            ann_dict.setdefault(image_id, []).append(ann)
        return image_dict, ann_dict

    @staticmethod
    def _generate_windows(sorted_image_ids, window_size: int, interval: int):
        """
        규칙:
          - 윈도우 인덱스: [s] + [s + m*interval - 1 for m=1..W-1], 각 위치는 [0, N-1]로 clip
          - 연속 윈도우의 시작 s는 step = max(1, interval-1)씩 증가
        """
        N = len(sorted_image_ids)
        if N == 0:
            return []

        step_start = max(1, interval - 1)
        windows = []
        for s_pos in range(0, N, step_start):
            pos_list = [s_pos] + [min(N - 1, s_pos + m * interval - 1) for m in range(1, window_size)]
            windows.append([sorted_image_ids[p] for p in pos_list])
        return windows

    @staticmethod
    def _check_window_files_exist(input_dir: str, window_image_ids: list) -> bool:
        return all(os.path.exists(os.path.join(input_dir, f"{img_id}.npy")) for img_id in window_image_ids)

    @staticmethod
    def _concat_window_frames(root: str, rid: str, image_ids: list, channel_indices: list) -> torch.Tensor:
        """
        연속 프레임을 채널축으로 concat → (C * W, H, W)
        """
        frames = []
        for img_id in image_ids:
            npy_path = os.path.join(root, f"{rid}.rep", f"{img_id}.npy")
            if not os.path.isfile(npy_path):
                raise FileNotFoundError(f"[CustomDataset] Missing input file: {npy_path}")

            arr = np.load(npy_path)
            arr = arr[channel_indices]
            if arr.ndim != 3:
                raise ValueError(f"[CustomDataset] Unexpected input shape: {arr.shape} at {npy_path}")
            frames.append(arr)

        return torch.from_numpy(np.concatenate(frames, axis=0))

    @staticmethod
    def _make_target_from_anns(anns: list, H: int, W: int, image_id: int) -> dict:
        """
        COCO bbox를 바이너리 mask로 바꿔 Mask R-CNN의 target 딕셔너리로 변환.
        """
        boxes, masks, labels, areas, iscrowd = [], [], [], [], []
        for ann in anns:
            x, y, w, h = map(int, ann["bbox"])
            if w <= 0 or h <= 0:
                continue

            x1, y1, x2, y2 = x, y, x + w, y + h
            # 이미지 경계 clip
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(0, min(x2, W))
            y2 = max(0, min(y2, H))
            if x2 <= x1 or y2 <= y1:
                continue

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
                "image_id": torch.tensor([image_id]),
                "area": torch.tensor(areas, dtype=torch.float32),
                "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, H, W), dtype=torch.uint8),
                "image_id": torch.tensor([image_id]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }
        return target

    # (선택) 정렬 키 유틸이 필요하면 유지
    @staticmethod
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r"(\d+)", os.path.basename(s))]
