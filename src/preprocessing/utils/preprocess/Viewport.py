import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from .viewport_parallel_utils import (
    preprocess_argmax_kernel_sum_parallel,
    preprocess_unique_local_maximums_parallel,
    preprocess_all_correct_parallel,
    save_single_result,
    read_single_csv
)
import traceback

ORIGIN_SHAPE = (128, 128)
KERNEL_SHAPE = (20, 12)
TILE_SIZE = 32
INTERVAL = 1


class Viewport:
    def __init__(self, replay_id):
        base_path = os.path.join(os.getcwd(), "data", "label")
        self.viewport_root = os.path.join(base_path, "src")
        self.result_root = os.path.join(base_path, "dst")
        self.replay_id = replay_id
        self.method = None
        self.vpds = []
        self.results = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def save(self):
        result_path = os.path.join(self.result_root, self.replay_id, self.method)
        from .viewport_parallel_utils import save_all_results
        try:
            save_all_results(results=self.results, path=result_path)
        except Exception as e:
            print("[Save Error]", e)
            import traceback
            traceback.print_exc()

    def load(self):
        vpds_paths = glob.glob(os.path.join(self.viewport_root, "*", f"{self.replay_id}.rep.vpd"))
        try:
            with Pool(cpu_count() // 2) as pool:
                self.vpds = list(pool.map(read_single_csv, vpds_paths))
            return bool(self.vpds)
        except Exception as e:
            print("[Load Error]", e)
            traceback.print_exc()
            return False

    def interpolation(self, dataframes):
        interpolated = []
        for df in dataframes:
            try:
                df = (df.set_index("frame")
                        .reindex(range(df["frame"].max()))
                        .ffill()
                        .reset_index()
                        .astype(int)
                        .set_index("frame"))
                interpolated.append(df)
            except Exception as e:
                print("[Interpolation Error]", e)
                traceback.print_exc()
        return interpolated

    def merge_dataframes(self, dataframes):
        try:
            num = len(dataframes)
            df = pd.concat(dataframes, axis=1).ffill().astype(int)
            df.columns = [f"vp{x}_{i + 1}" for i in range(num) for x in ("x", "y")]
            df = (df / TILE_SIZE).astype(int).reset_index()
            return df, num
        except Exception as e:
            print("[Merge Error]", e)
            traceback.print_exc()
            return pd.DataFrame(), 0

    def preprocess_argmax_kernel_sum(self, dataframe, num_vpds):
        return preprocess_argmax_kernel_sum_parallel(
            dataframe, num_vpds, KERNEL_SHAPE, ORIGIN_SHAPE, INTERVAL
        )

    def preprocess_unique_local_maximums(self, dataframe, num_vpds):
        from local_peaks import get_local_maximums, get_unique_peaks2
        return preprocess_unique_local_maximums_parallel(
            dataframe, num_vpds, KERNEL_SHAPE, ORIGIN_SHAPE, INTERVAL,
            get_local_maximums, get_unique_peaks2
        )

    def preprocess_all_correct(self, dataframe, num_vpds):
        return preprocess_all_correct_parallel(
            dataframe, num_vpds, INTERVAL
        )

    def preprocess_consider_previous(self, dataframe: pd.DataFrame, num_vpds: int):
        def distance_2d(p1, p2):
            return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

        def compare_with_previous(channel_kernel_sum, unique_peaks, previous):
            if previous is None:
                return unique_peaks[np.random.choice(len(unique_peaks))]
            dists = [distance_2d(p, previous) for p in unique_peaks]
            return unique_peaks[np.argmin(dists)]

        from local_peaks import get_local_maximums, get_unique_peaks
        result = []
        previous_viewport = None

        for t in tqdm(range(0, len(dataframe), INTERVAL), desc="Processing viewport(consider previous)"):
            try:
                df_t = dataframe.loc[dataframe["frame"] == t].squeeze()
                channel = np.zeros(ORIGIN_SHAPE)
                kernel = np.ones(KERNEL_SHAPE)
                for i in range(num_vpds):
                    x = int(df_t[f"vpx_{i + 1}"])
                    y = int(df_t[f"vpy_{i + 1}"])
                    channel[x:x + KERNEL_SHAPE[0], y:y + KERNEL_SHAPE[1]] += kernel
                channel = channel.T

                width_tile = ORIGIN_SHAPE[0] - KERNEL_SHAPE[0]
                height_tile = ORIGIN_SHAPE[1] - KERNEL_SHAPE[1]
                kernel_sum = np.zeros((width_tile, height_tile))
                for x in range(width_tile):
                    for y in range(height_tile):
                        kernel_sum[x][y] = channel[x:x + KERNEL_SHAPE[0], y:y + KERNEL_SHAPE[1]].sum()

                peaks, _ = get_local_maximums(kernel_sum)
                unique_peaks = get_unique_peaks(peaks)

                current = compare_with_previous(kernel_sum, unique_peaks, previous_viewport)
                previous_viewport = current
                result.append(current)
            except Exception as e:
                print(f"[Error at frame {t}]", e)
                traceback.print_exc()
        return result

    def run(self, method):
        print(f"[Viewport] Method selected: {method}")
        try:
            print("[Viewport] Interpolating dataframes...")
            dataframes = self.interpolation(self.vpds)

            print("[Viewport] Merging dataframes...")
            dataframe, num_vpds = self.merge_dataframes(dataframes)

            methods = {
                "legacy": self.preprocess_argmax_kernel_sum,
                "unique_local_maximums": self.preprocess_unique_local_maximums,
                "all_correct": self.preprocess_all_correct,
                "consider_previous": self.preprocess_consider_previous,
            }

            if method not in methods:
                raise NotImplementedError(f"Method '{method}' is not implemented.")

            self.method = method
            print(f"[Viewport] Running preprocessing: {method}")
            self.results = methods[method](dataframe, num_vpds)
            print(f"[Viewport] Finished preprocessing with method: {method}")
        except Exception as e:
            print("[Viewport.run] Exception occurred")
            traceback.print_exc()