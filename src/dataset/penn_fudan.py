import os
import numpy as np
import torch

class PennFudanDataset:
    def __init__(self, path, transforms, window_size, training, mode="default", vpds_base_path="/home/joo/Downloads/argmax_kernel_sum_vpds"):
        self.root = path
        self.transforms = transforms
        self.window_size = window_size
        self.tr_replay = training
        self.mode = mode
        self.vpds_base_path = vpds_base_path

        self.dir_paths = [
            os.path.join(path, d) + '/'
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d)) and d in map(str, training)
        ]

        self.seq_indexs = []
        index_a = 0
        for i, dir_path in enumerate(self.dir_paths):
            seq_len = len(os.listdir(dir_path)) - 150
            self.seq_indexs.append((i, index_a, index_a + seq_len))
            index_a += seq_len

    def __len__(self):
        return self.seq_indexs[-1][-1]

    def __getitem__(self, idx):
        for i, start, end in self.seq_indexs:
            if start <= idx < end:
                real_idx = idx - start
                base_path = self.dir_paths[i]

                data = [
                    np.load(os.path.join(base_path, f"{real_idx + 147 + j}.npy"), allow_pickle=True)[0]
                    for j in range(4)
                ]
                npy150 = np.load(os.path.join(base_path, f"{real_idx + 150}.npy"), allow_pickle=True)
                masks, boxes = self._process_labels(i, base_path, real_idx, npy150)

                input_data = self.preprocessing(*data)
                target = self._build_target(idx, masks, boxes, i, base_path, real_idx)
                return input_data, target

        raise IndexError(f"Index {idx} out of range")

    def preprocessing(self, *data):
        height, width = data[0].shape[1:]
        temp = np.zeros((self.window_size, 9, height, width))
        temp[:4] = data
        reshaped = temp.reshape(self.window_size * 9, height, -1)
        return torch.FloatTensor(reshaped)

    def _process_labels(self, i, base_path, real_idx, npy150):
        masks, boxes = npy150[1], npy150[2]

        if self.mode == "point2_labels" and not boxes:
            return self._fallback_valid_data(i, real_idx + 3)[1:]

        if self.mode == "one":
            masks = masks[i % 5: i % 5 + 1]

        if self.mode == "point":
            replay_name = os.path.basename(os.path.normpath(base_path))
            coor_path = os.path.join(self.vpds_base_path, replay_name, "argmax_kernel_sum", f"{real_idx + 150}.vpds.npy")
            coor_label = np.load(coor_path)
            return masks, self._boxes_from_masks_with_point(masks, coor_label)

        return masks, boxes

    def _build_target(self, idx, masks, boxes, i, base_path, real_idx):
        if self.mode == "point":
            labels = torch.LongTensor([1, 1, 1, 1, 1, 2])
        else:
            num_objs = len(masks) if self.mode != "one" else 1
            labels = torch.ones((num_objs,), dtype=torch.int64)
            if self.mode != "point2_labels":
                boxes = self.masks_to_boxes(masks[:num_objs])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        return {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

    def _fallback_valid_data(self, i, real_idx):
        while True:
            npy150 = np.load(os.path.join(self.dir_paths[i], f"{real_idx + 150}.npy"), allow_pickle=True)
            if npy150[2]:
                break
            real_idx += 1

        data = [
            np.load(os.path.join(self.dir_paths[i], f"{real_idx + 147 + j}.npy"), allow_pickle=True)[0]
            for j in range(4)
        ]
        return data, npy150[1], npy150[2]

    def masks_to_boxes(self, masks):
        boxes = []
        for mask in masks:
            pos = np.where(mask)
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            boxes.append([xmin, ymin, xmax, ymax])
        return boxes

    def _boxes_from_masks_with_point(self, masks, coor_label):
        boxes = []
        for j in range(6):
            if j == 5:
                y, x = coor_label
                boxes.append([x, y, x + 19, y + 11])
            else:
                pos = np.where(masks[j])
                ymin, ymax = np.min(pos[0]), np.max(pos[0])
                xmin, xmax = np.min(pos[1]), np.max(pos[1])
                boxes.append([xmin, ymin, xmax, ymax])
        return boxes
