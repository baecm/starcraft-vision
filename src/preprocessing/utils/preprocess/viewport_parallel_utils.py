import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm
import traceback

def group_by_frame(dataframe):
    return dataframe.groupby("frame")

def process_single_frame_argmax(args):
    t, df_row, num_vpds, kernel_shape, origin_shape = args
    try:
        channel = np.zeros(origin_shape)
        kernel = np.ones(kernel_shape)
        for i in range(num_vpds):
            x = int(df_row[f"vpx_{i + 1}"])
            y = int(df_row[f"vpy_{i + 1}"])
            channel[x:x + kernel_shape[0], y:y + kernel_shape[1]] += kernel
        channel = channel.T

        width_tile = origin_shape[0] - kernel_shape[0]
        height_tile = origin_shape[1] - kernel_shape[1]
        kernel_sum = np.zeros((width_tile, height_tile))
        for x in range(width_tile):
            for y in range(height_tile):
                kernel_sum[x][y] = channel[x:x + kernel_shape[0], y:y + kernel_shape[1]].sum()

        max_index = np.argmax(kernel_sum)
        x_tile = max_index % kernel_sum.shape[1]
        y_tile = max_index // kernel_sum.shape[1]

        return t, (x_tile, y_tile)
    except Exception as e:
        print(f"[Worker Error @ frame {t}] {e}")
        traceback.print_exc()
        return t, None

def preprocess_argmax_kernel_sum_parallel(dataframe: pd.DataFrame, num_vpds: int, kernel_shape, origin_shape, interval: int):
    frame_indices = list(range(0, len(dataframe), interval))
    grouped = group_by_frame(dataframe)
    tasks = [(t, grouped.get_group(t).reset_index(drop=True).iloc[0], num_vpds, kernel_shape, origin_shape) for t in frame_indices]

    with Pool(cpu_count() // 2) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_frame_argmax, tasks, chunksize=1), total=len(tasks), desc="Processing viewport (legacy)"))

    results = sorted([r for r in results if r[1] is not None], key=lambda x: x[0])
    return [r[1] for r in results]

def process_local_max_frame(args):
    t, df_row, num_vpds, kernel_shape, origin_shape, get_local_maximums, get_unique_peaks2 = args
    try:
        channel = np.zeros(origin_shape)
        kernel = np.ones(kernel_shape)
        for i in range(num_vpds):
            x = int(df_row[f"vpx_{i + 1}"])
            y = int(df_row[f"vpy_{i + 1}"])
            channel[x:x + kernel_shape[0], y:y + kernel_shape[1]] += kernel
        channel = channel.T

        width_tile = origin_shape[0] - kernel_shape[0]
        height_tile = origin_shape[1] - kernel_shape[1]
        kernel_sum = np.zeros((width_tile, height_tile))
        for x in range(width_tile):
            for y in range(height_tile):
                kernel_sum[x][y] = channel[x:x + kernel_shape[0], y:y + kernel_shape[1]].sum()

        peaks, _ = get_local_maximums(kernel_sum)
        unique_peaks = get_unique_peaks2(peaks)
        return t, unique_peaks
    except Exception as e:
        print(f"[Worker Error @ frame {t}] {e}")
        traceback.print_exc()
        return t, None

def preprocess_unique_local_maximums_parallel(dataframe: pd.DataFrame, num_vpds: int, kernel_shape, origin_shape, interval: int, get_local_maximums, get_unique_peaks2):
    frame_indices = list(range(0, len(dataframe), interval))
    grouped = group_by_frame(dataframe)
    tasks = [
        (t, grouped.get_group(t).reset_index(drop=True).iloc[0], num_vpds, kernel_shape, origin_shape, get_local_maximums, get_unique_peaks2)
        for t in frame_indices
    ]

    with Pool(cpu_count() // 2) as pool:
        results = list(tqdm(pool.imap_unordered(process_local_max_frame, tasks, chunksize=1), total=len(tasks), desc="Processing viewport (local maximums)"))

    results = sorted([r for r in results if r[1] is not None], key=lambda x: x[0])
    return [r[1] for r in results]

def process_all_correct_frame(args):
    t, df_t, num_vpds = args
    labels = np.split(np.asarray(df_t.set_index("frame")).squeeze(), num_vpds)
    return t, labels

def preprocess_all_correct_parallel(dataframe: pd.DataFrame, num_vpds: int, interval: int):
    grouped = group_by_frame(dataframe)
    frames = [(t, grouped.get_group(t), num_vpds) for t in range(0, len(dataframe), interval)]
    with Pool(processes=cpu_count() // 2) as pool:
        result = list(tqdm(pool.imap_unordered(process_all_correct_frame, frames, chunksize=1), total=len(frames), desc="Processing viewport (all correct)"))
    result.sort(key=lambda x: x[0])
    return [r[1] for r in result]

def save_single_result(args):
    t, result, path = args
    try:
        np.save(os.path.join(path, f"{t}.vpds.npy"), result)
    except Exception as e:
        print(f"[Save Error @ {t}] {e}")
        traceback.print_exc()

def save_all_results(results, path):
    os.makedirs(path, exist_ok=True)
    args = [(t, result, path) for t, result in enumerate(results)]
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        list(tqdm(executor.map(lambda x: save_single_result(x), args), total=len(args), desc="Saving results"))

def read_single_csv(path):
    try:
        return pd.read_csv(path, index_col=None)
    except Exception as e:
        print(f"[CSV Read Error @ {path}] {e}")
        traceback.print_exc()
        return pd.DataFrame()
