import os
import numpy as np
import tqdm
import glob
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Generator

import code.src.config as config

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--replays", type=int, nargs="+", required=True)
    parser.add_argument("--method", type=str, default=config.LABEL_METHODS[0], choices=config.LABEL_METHODS)
    parser.add_argument("--output", type=str, default=config.OUTPUT_TYPES[1], choices=config.OUTPUT_TYPES)
    parser.add_argument("--result-dir", type=str, default=os.path.join(os.getcwd(), "data", "pair"))
    return parser.parse_args()


class LazyNpyLoader:
    def __init__(self, files: List[str]):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        return np.load(self.files[idx], allow_pickle=True)


class LazyScatterLabels:
    def __init__(self, loader: LazyNpyLoader, kernel_shape=config.KERNEL_SHAPE, origin_shape=config.ORIGIN_SHAPE):
        self.loader = loader
        self.kernel = np.ones(kernel_shape)
        self.kh, self.kw = kernel_shape
        self.oh, self.ow = origin_shape
        self._cache = {}

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, idx: int):
        if idx in self._cache:
            return self._cache[idx]

        labels = self.loader[idx]
        channels = []
        for label in labels:
            if isinstance(label, np.ndarray) and label.shape == (2,):
                x, y = label
                channel = np.zeros((self.oh, self.ow))
                channel[x:x + self.kh, y:y + self.kw] += self.kernel
                channels.append(channel.T)
            elif isinstance(label, (int, np.int64)):
                x = label // self.ow
                y = label % self.ow
                channel = np.zeros((self.oh, self.ow))
                channel[x:x + self.kh, y:y + self.kw] += self.kernel
                channels.append(channel.T)
            else:
                print(f"[Error] Invalid label format: {label}")

        self._cache[idx] = channels
        return channels


def get_sorted_files(directory: str) -> List[str]:
    files = glob.glob(os.path.join(directory, "*.npy"))
    valid_files = []
    for f in files:
        filename = os.path.basename(f)
        basename = filename.replace(".vpds", "")
        name, _ = os.path.splitext(basename)
        try:
            int(name)
            valid_files.append(f)
        except ValueError:
            continue
    return sorted(valid_files, key=lambda f: int(os.path.splitext(os.path.basename(f).replace(".vpds", ""))[0]))


def load_input(replay: int, show_progress=False) -> LazyNpyLoader:
    path = os.path.join("data", "input", "dst", f"{replay}.rep")
    files = get_sorted_files(path)
    if not files:
        print(f"[Warning] No input found at {path}")
    if show_progress:
        print(f"[Info] Loading {len(files)} input frames...")
        for _ in tqdm.tqdm(files, desc="Loading frames", miniters=max(1, len(files) // 20)):
            pass
    return LazyNpyLoader(files)


def load_labels(replay: int, method: str, output: str, show_progress=False):
    path = os.path.join("data", "label", "dst", f"{replay}.rep", method)
    files = get_sorted_files(path)
    if not files:
        print(f"[Warning] No labels found at {path}")
        return None
    loader = LazyNpyLoader(files)
    if output == "coord":
        return loader
    elif output == "channel":
        scatter_loader = LazyScatterLabels(loader)
        if show_progress:
            print(f"[Info] Scattering {len(scatter_loader)} labels...")
            for _ in tqdm.tqdm(range(len(scatter_loader)), desc="Scattering", miniters=max(1, len(scatter_loader) // 20)):
                _ = scatter_loader[_]
        return scatter_loader
    return None


def make_pairs(inputs, labels) -> Tuple[Generator, int]:
    if labels is None:
        return [], 0
    length = min(len(inputs), len(labels))
    pairs = []
    for i in range(length):
        if isinstance(inputs[i], np.ndarray) and isinstance(labels[i], list):
            pairs.append((inputs[i], labels[i]))
        else:
            print(f"[Error] Invalid input-label pair at index {i}: {inputs[i]}, {labels[i]}")
    return pairs, length


def save_pair(args: Tuple[int, Tuple[np.ndarray, np.ndarray], str]):
    idx, pair, path = args
    try:
        if not isinstance(pair, tuple) or len(pair) != 2:
            print(f"[Error] Invalid pair format at index {idx}: {pair}")
            return
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, f"{idx}.npy"), np.array(pair, dtype=object))
    except Exception as e:
        print(f"[Error] Failed to save index {idx}: {e}")


def store_pairs(pairs, count: int, result_dir: str, replay: int, method: str):
    path = os.path.join(result_dir, f"{replay}.rep", method)
    args = [(i, pair, path) for i, pair in enumerate(pairs)]
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(save_pair, arg) for arg in args]
        for _ in tqdm.tqdm(as_completed(futures), total=count, desc=f"Saving replay {replay}", miniters=max(1, count // 20)):
            pass


def process_replay(replay: int, method: str, output: str, result_dir: str):
    print(f"\n[Start] Processing replay {replay}")
    print(f"[INFO] Output type: {output}")

    try:
        inputs = load_input(replay, show_progress=True)
        if len(inputs) == 0:
            print(f"[Skip] Replay {replay}: no input frames found")
            return

        labels = load_labels(replay, method, output, show_progress=True)
        if labels is None or len(labels) == 0:
            print(f"[Skip] Replay {replay}: no labels found")
            return

        print(f"[Info] Loaded {len(inputs)} input frames")
        print(f"[Info] Loaded {len(labels)} labels")

        pairs, count = make_pairs(inputs, labels)
        if count == 0:
            print(f"[Skip] Replay {replay}: no valid frame-label pairs")
            return

        store_pairs(pairs, count, result_dir, replay, method)

    except Exception as e:
        print(f"[Error] Replay {replay} failed: {e}")


def main():
    args = parse_arguments()
    for replay in args.replays:
        process_replay(replay, args.method, args.output, args.result_dir)


if __name__ == "__main__":
    main()
