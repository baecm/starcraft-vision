import os
import numpy as np
import torch


import os
import numpy as np
import torch
from .penn_fudan import PennFudanDataset as BasePennFudanDataset


class CustomPennFudanDataset(BasePennFudanDataset):
    def __init__(self, replays, training, window_size):
        # 원본 root/transform 대신 custom 파라미터 사용
        self.dir_paths = [
            os.path.join(replays, dir_name) + "/"
            for dir_name in os.listdir(replays)
            if os.path.isdir(os.path.join(replays, dir_name)) and dir_name in map(str, training)
        ]
        self.window_size = window_size
        self.seq_indexs = self._generate_sequence_indices()

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
        return self.seq_indexs[-1][-1]

    def __getitem__(self, idx):
        for dir_path, start, end in self.seq_indexs:
            if start <= idx < end:
                real_idx = idx - start
                return self._load_data(dir_path, real_idx)

    def _load_data(self, dir_path, real_idx):
        # Load data and masks
        frames = [np.load(os.path.join(dir_path, f"{real_idx + i}.npy"), allow_pickle=True)[0] for i in range(self.window_size)]
        masks = np.load(os.path.join(dir_path, f"{real_idx + 150}.npy"), allow_pickle=True)[1]

        # Preprocess the data
        input_data = self.preprocessing(*frames)

        # Extract bounding boxes from masks
        boxes = self._extract_boxes(masks)

        # Prepare the target dictionary
        target = self._create_target(boxes, masks, real_idx)

        return input_data, target

    def _extract_boxes(self, masks):
        boxes = []
        for mask in masks:
            pos = np.where(mask)
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        return torch.tensor(boxes, dtype=torch.float32)

    def _create_target(self, boxes, masks, idx):
        num_objs = len(boxes)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

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
        temp = np.zeros([self.window_size, 9, data[0].shape[1], data[0].shape[2]])
        for i, d in enumerate(data):
            temp[i] = d

        data = temp.reshape(self.window_size * temp.shape[1], temp.shape[2], -1)
        return torch.FloatTensor(data)
