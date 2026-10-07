"""
The training dataset: windows of consecutive game-state frames, with the
observers' viewports (or the ranked attention modes) as box targets.

One sample is
  input   the frames of one window stacked on the channel axis,
          (C * window_size, H, W), read from <input_root>/<rid>.rep/<frame>.npy
  target  Mask R-CNN-style boxes / masks / labels for the window's LAST frame,
          from <label_root>/<rid>.rep/<label_method>.pkl (COCO images and
          annotations), plus the frame's ranked modes when the mode cache is
          loaded, and the next window when pair_mode is on (for L_smooth).

A window is the frame ids at positions [s, s+interval-1, s+2*interval-1, ...]
of the replay's sorted frames (window_size of them), and consecutive windows
start interval-1 frames apart; see _generate_windows. Windows with a missing
input file are dropped.
"""
import os
import pickle

import numpy as np
import torch

import config
from metrics.modes import box_from_center, box_slices
from utils.logger import Logger

# What the box targets are:
#   none - the observers' own viewports, one box per observer (the default)
#   hard - one viewport-sized box per ranked attention mode, every box label 1
#   soft - the same boxes, plus each box's support / U as "mode_weight", which
#          a detector can use as a graded class target (see
#          models/backbones/maskrcnn.py, SupportTargetRoIHeads)
MODE_TARGET_CHOICES = ("none", "hard", "soft")


class StarCraftWindowDataset(torch.utils.data.Dataset):
    """Windows of one or more replays; see the module docstring.

    use_mode_cache loads the ranked-mode cache (dataset/mode_cache.py) and
    puts each frame's modes in target["modes"]. If the cache file is missing,
    that replay gets no modes and only a warning is logged; Director-CenterNet
    then trains with empty mode targets. train.py builds the cache before the
    dataset with the same (default) extraction parameters, so this only happens
    when those parameters are overridden.
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
        trim_tail: int = 0,
        pair_mode: bool = False,
        use_mode_cache: bool = True,
        mode_targets: str = "none",
    ):
        if mode_targets not in MODE_TARGET_CHOICES:
            raise ValueError(f"mode_targets must be one of {MODE_TARGET_CHOICES}, got {mode_targets!r}")
        # Box targets from the ranked modes need the mode cache, whatever the
        # caller asked for.
        if mode_targets != "none":
            use_mode_cache = True
        self.mode_targets = mode_targets
        self.input_root = input_root
        self.label_root = label_root
        self.label_method = label_method
        self.training = training
        self.verbose = verbose
        self.window_size = int(window_size)
        self.interval = max(1, int(interval))
        self.trim_tail = max(0, int(trim_tail))
        self.pair_mode = bool(pair_mode)
        self.use_mode_cache = bool(use_mode_cache)

        self.channel_indices = self._build_channel_indices(include_components)

        # one tuple per sample:
        # (rid, window frame ids, next window frame ids, has_next, image_dict, ann_dict)
        self.files = []
        self.mode_caches = {}

        for rid in map(str, training_ids):
            input_dir = os.path.join(self.input_root, f"{rid}.rep")
            pkl_path = os.path.join(label_root, f"{rid}.rep", f"{label_method}.pkl")

            if not (os.path.isdir(input_dir) and os.path.isfile(pkl_path)):
                if self.verbose:
                    Logger.warn(f"Skipping {rid}: missing input dir or pickle file.")
                continue

            if self.use_mode_cache:
                try:
                    from .mode_cache import load_mode_cache
                    self.mode_caches[rid] = load_mode_cache(label_root, rid, label_method)
                except Exception as e:
                    if self.verbose:
                        Logger.warn(f"Mode cache not loaded for {rid}: {e}")
                    self.mode_caches[rid] = None

            image_dict, ann_dict = self._load_meta_from_pickle(pkl_path)

            sorted_image_ids = sorted(image_dict.keys())
            if self.trim_tail > 0 and len(sorted_image_ids) > self.trim_tail:
                sorted_image_ids = sorted_image_ids[:-self.trim_tail]

            windows = self._generate_windows(sorted_image_ids, self.window_size, self.interval)
            # One listdir per replay rather than one stat per file: the data
            # sits on a network share, where per-file calls dominate.
            existing_files = set(os.listdir(input_dir))
            valid_windows = [win for win in windows if all(f"{img_id}.npy" in existing_files for img_id in win)]

            for i, win in enumerate(valid_windows):
                # the last window of a replay has no successor: it pairs with
                # itself so the tensor shapes stay uniform, but has_next=False
                # keeps that zero-displacement pair out of L_smooth.
                has_next = i + 1 < len(valid_windows)
                next_win = valid_windows[i + 1] if has_next else win
                self.files.append((rid, win, next_win, has_next, image_dict, ann_dict))

        if not self.files:
            raise RuntimeError("Empty dataset or no valid windows found.")

        if indices is not None:
            self.files = [self.files[i] for i in indices]

        if self.verbose:
            Logger.info(f"Total windows (samples): {len(self.files)} (pair_mode={self.pair_mode}, use_mode_cache={self.use_mode_cache})")
            value_to_name_map = {member.value: name for name, member in config.Channel.__members__.items()}
            channel_names = [value_to_name_map[i] for i in self.channel_indices]
            Logger.info(f"Using {len(self.channel_indices)} channels: {channel_names}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rid, window_image_ids, next_window_image_ids, has_next, image_dict, ann_dict = self.files[idx]

        input_tensor = self._concat_window_frames(
            root=self.input_root,
            rid=rid,
            image_ids=window_image_ids,
            channel_indices=self.channel_indices
        ).float()
        _, H, W = input_tensor.shape

        # the target is the window's last frame
        target_img_id = window_image_ids[-1]
        if target_img_id not in image_dict:
            raise KeyError(f"[StarCraftWindowDataset] Target Image ID {target_img_id} not found in image_dict")

        m_info = None
        if self.use_mode_cache and self.mode_caches.get(rid) is not None:
            m_info = self.mode_caches[rid].get(target_img_id)

        if self.mode_targets == "none":
            anns = ann_dict.get(target_img_id, [])
            target = self._make_target_from_anns(anns, H, W, target_img_id)
        else:
            if self.mode_caches.get(rid) is None:
                raise RuntimeError(f"[StarCraftWindowDataset] mode_targets={self.mode_targets!r} but replay {rid} has no mode cache")
            # A frame missing from the cache had no observer boxes, so it has
            # no modes either: an empty target, as with mode_targets=none.
            target = self._make_target_from_modes(m_info, H, W, target_img_id)

        if m_info is not None:
            target["modes"] = {
                "centers": torch.from_numpy(m_info["centers"]),
                "support": torch.from_numpy(m_info["support"]),
                "n_observers": m_info.get("n_observers", 5),
            }

        # the next window, for L_smooth
        if self.pair_mode:
            next_input_tensor = self._concat_window_frames(
                root=self.input_root,
                rid=rid,
                image_ids=next_window_image_ids,
                channel_indices=self.channel_indices
            ).float()
            target["next_image"] = next_input_tensor
            target["next_valid"] = bool(has_next)

            next_target_img_id = next_window_image_ids[-1]
            if self.use_mode_cache and self.mode_caches.get(rid) is not None:
                next_m_info = self.mode_caches[rid].get(next_target_img_id)
                if next_m_info is not None:
                    target["next_modes"] = {
                        "centers": torch.from_numpy(next_m_info["centers"]),
                        "support": torch.from_numpy(next_m_info["support"]),
                        "n_observers": next_m_info.get("n_observers", 5),
                    }

        # Tracing metadata. The models ignore it; the training loop only moves
        # tensors to the device.
        target["rid"] = str(rid)
        target["window_image_ids"] = [int(x) for x in window_image_ids]
        target["sample_idx"] = int(idx)

        return input_tensor, target

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_channel_indices(include_components):
        """Input channel indices for the named components (all channels if None)."""
        if include_components:
            expanded_components = []
            for c in include_components:
                c_clean = str(c).lower().strip()
                if c_clean in ["units", "unit"]:
                    expanded_components.extend(["worker", "ground", "air"])
                elif c_clean in ["buildings", "building"]:
                    expanded_components.append("building")
                elif c_clean in ["resources", "resource"]:
                    expanded_components.append("resource")
                elif c_clean in config.COMPONENT_CHANNEL_MAP:
                    expanded_components.append(c_clean)
                else:
                    Logger.warn(f"Unknown component '{c}' ignored. Available: {list(config.COMPONENT_CHANNEL_MAP.keys())}")

            indices = set()
            for c in expanded_components:
                indices.update(config.COMPONENT_CHANNEL_MAP.get(c, []))
            return sorted(list(indices)) if indices else list(range(len(config.Channel)))
        return list(range(len(config.Channel)))

    @staticmethod
    def _load_meta_from_pickle(pkl_path: str):
        """(frame id -> image record, frame id -> [annotations]) from a label pickle."""
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
        Window positions are [s] + [s + m*interval - 1 for m = 1..window_size-1],
        each clipped to [0, N-1], and consecutive windows start
        max(1, interval-1) positions apart.
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
    def _concat_window_frames(root: str, rid: str, image_ids: list, channel_indices: list) -> torch.Tensor:
        """The window's frames stacked on the channel axis: (C * window_size, H, W)."""
        frames = []
        for img_id in image_ids:
            npy_path = os.path.join(root, f"{rid}.rep", f"{img_id}.npy")
            if not os.path.isfile(npy_path):
                raise FileNotFoundError(f"[StarCraftWindowDataset] Missing input file: {npy_path}")

            arr = np.load(npy_path, mmap_mode="r")
            arr = arr[channel_indices]
            if arr.ndim != 3:
                raise ValueError(f"[StarCraftWindowDataset] Unexpected input shape: {arr.shape} at {npy_path}")
            frames.append(arr)

        return torch.from_numpy(np.concatenate(frames, axis=0))

    @staticmethod
    def _make_target_from_anns(anns: list, H: int, W: int, image_id: int) -> dict:
        """
        The observers' viewports (COCO bbox [x, y, w, h], top-left corner) as a
        Mask R-CNN target: xyxy boxes clipped to the map, and filled-box masks.
        """
        boxes, masks, labels, areas, iscrowd = [], [], [], [], []
        for ann in anns:
            x, y, w, h = map(int, ann["bbox"])
            if w <= 0 or h <= 0:
                continue

            x1, y1, x2, y2 = x, y, x + w, y + h
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

    @staticmethod
    def _make_target_from_modes(m_info, H: int, W: int, image_id: int) -> dict:
        """
        One viewport-sized box per ranked attention mode, in the same target
        format as _make_target_from_anns.

        This is the control for the question whether a proposal detector's
        redundant and empty boxes come from its supervision: five observer
        viewports put up to five near-identical boxes on one mode, whereas
        modes are at least MODE_EXTRACTION_MIN_SEP tiles apart, so here each
        mode is one box. The box is centered on the mode and shifted (not
        cropped) to stay inside the map, as the evaluation's canonical mode
        region is (metrics.modes.box_from_center).

        "mode_weight" is support / U per box. It is a target field the
        standard RoIHeads ignores; SupportTargetRoIHeads reads it.
        """
        centers = np.zeros((0, 2)) if m_info is None else np.asarray(m_info["centers"], dtype=float)
        support = np.zeros(0) if m_info is None else np.asarray(m_info["support"], dtype=float)
        n_obs = 1 if m_info is None else max(1, int(m_info.get("n_observers", config.NUM_OBSERVERS_U)))

        boxes, masks, weights = [], [], []
        for center, s in zip(centers, support):
            box = box_from_center(center, config.VIEWPORT_SIZE_HW, H, W)
            ys, xs = box_slices(box, H, W)
            if ys.stop <= ys.start or xs.stop <= xs.start:
                continue
            m = np.zeros((H, W), dtype=np.uint8)
            m[ys, xs] = 1
            boxes.append([xs.start, ys.start, xs.stop, ys.stop])
            masks.append(torch.from_numpy(m))
            weights.append(float(s) / n_obs)

        n = len(boxes)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(n, 4),
            "labels": torch.ones((n,), dtype=torch.int64),
            "masks": torch.stack(masks) if n else torch.zeros((0, H, W), dtype=torch.uint8),
            "image_id": torch.tensor([image_id]),
            "area": torch.tensor([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes], dtype=torch.float32),
            "iscrowd": torch.zeros((n,), dtype=torch.int64),
            "mode_weight": torch.tensor(weights, dtype=torch.float32),
        }
        return target
