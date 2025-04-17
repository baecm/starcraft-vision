import os
import numpy as np
import torch
from .penn_fudan import PennFudanDataset as BasePennFudanDataset
from utils.logger import Logger


class CustomPennFudanDataset(BasePennFudanDataset):
    def __init__(self, replays, training, window_size):
        Logger.info("[CustomDataset] Initializing CustomPennFudanDataset...")

        self.dir_paths = [
            os.path.join(replays, dir_name) + "/"
            for dir_name in os.listdir(replays)
            if os.path.isdir(os.path.join(replays, dir_name)) and dir_name in map(str, training)
        ]
        Logger.info(f"[CustomDataset] Using {len(self.dir_paths)} replay directories: {self.dir_paths}")

        self.window_size = window_size
        self.seq_indexs = self._generate_sequence_indices()

        Logger.info(f"[CustomDataset] Total sequence segments: {len(self.seq_indexs)}")
        Logger.debug(f"[CustomDataset] Sequence index map: {self.seq_indexs}")

    def _generate_sequence_indices(self):
        seq_indexs = []
        index_a = 0
        for dir_path in self.dir_paths:
            num_files = len(os.listdir(dir_path))
            valid_length = num_files - 150  # usable frame 수
            index_b = index_a + valid_length
            seq_indexs.append((dir_path, index_a, index_b))
            index_a = index_b
        return seq_indexs

    def __len__(self):
        total_length = self.seq_indexs[-1][-1]
        Logger.debug(f"[CustomDataset] __len__ called, total length: {total_length}")
        return total_length

    def __getitem__(self, idx):
        for dir_path, start, end in self.seq_indexs:
            if start <= idx < end:
                real_idx = idx - start
                Logger.debug(f"[CustomDataset] Fetching idx={idx} from {dir_path}, real_idx={real_idx}")
                return self._load_data(dir_path, real_idx)

        Logger.error(f"[CustomDataset] Index {idx} out of range!")
        raise IndexError(f"Index {idx} out of range in CustomPennFudanDataset")

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
