import os
import numpy as np
import torch
from torch.utils.data import Dataset

import config
from utils.logger import Logger

class InferenceDataset(Dataset):
    """
    A custom dataset for Mask R-CNN inference that reads input .npy files with windowing.
    - Assumes that frames for each replay_id exist in 'data/input/dst/{replay_id}.rep/*.npy'.
    - Creates sliding windows of size `window_size`.
    - __getitem__ returns a tuple: (image_tensor, (replay_id, frame_id)).
      - The image_tensor is a stack of frames in the window (concatenated along channel axis).
      - The frame_id corresponds to the *last* frame in the window.
    """
    def __init__(self, input_root: str, replay_ids: list, window_size: int = 1, include_components: list = None):
        super().__init__()
        self.input_root = input_root
        self.window_size = window_size

        if include_components:
            self.channel_indices = sorted(sum([config.COMPONENT_CHANNEL_MAP[c] for c in include_components], []))
        else:
            self.channel_indices = list(range(len(config.Channel)))

        value_to_name_map = {member.value: name for name, member in config.Channel.__members__.items()}
        channel_names = [value_to_name_map[i] for i in self.channel_indices]
        Logger.info(f"[InferenceDataset] Using {len(self.channel_indices)} channels: {channel_names}")
        Logger.info(f"[InferenceDataset] Using window size: {self.window_size}")

        # Create a list of (replay_id, target_frame_id, [list_of_npy_paths_in_window]) tuples.
        self.indexes = []
        for rid in map(str, replay_ids):
            rep_dir = os.path.join(self.input_root, f"{rid}.rep")
            if not os.path.isdir(rep_dir):
                Logger.warn(f"[InferenceDataset] Missing directory: {rep_dir}")
                continue

            # Sort all .npy files numerically.
            npy_files = sorted(
                [f for f in os.listdir(rep_dir) if f.endswith(".npy")],
                key=lambda s: int(os.path.splitext(s)[0])
            )

            # Create sliding windows
            if len(npy_files) >= self.window_size:
                for i in range(len(npy_files) - self.window_size + 1):
                    window_files = npy_files[i: i + self.window_size]
                    window_paths = [os.path.join(rep_dir, f) for f in window_files]
                    target_frame_id = int(os.path.splitext(window_files[-1])[0])
                    self.indexes.append((rid, target_frame_id, window_paths))

        if len(self.indexes) == 0:
            raise RuntimeError(
                f"No .npy files or valid windows found for replays {replay_ids} "
                f"with window size {self.window_size}. Aborting."
            )

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        rid, target_frame_id, window_paths = self.indexes[idx]

        window_frames = []
        for npy_path in window_paths:
            arr = np.load(npy_path)
            arr = arr[self.channel_indices]
            if arr.ndim != 3:
                raise ValueError(f"Unexpected array shape {arr.shape} at {npy_path}")
            window_frames.append(arr)

        # Concatenate frames along the channel axis (C * window, H, W)
        img = torch.from_numpy(np.concatenate(window_frames, axis=0)).float()

        return img, (rid, target_frame_id)