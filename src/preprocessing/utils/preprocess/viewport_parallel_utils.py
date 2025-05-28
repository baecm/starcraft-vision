import os
import tqdm
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import traceback
import json
import config


def compute_kernel_sum(channel, kernel_shape):
    width_tile = channel.shape[0] - kernel_shape[0]
    height_tile = channel.shape[1] - kernel_shape[1]
    kernel_sum = np.zeros((width_tile, height_tile))
    for x in range(width_tile):
        for y in range(height_tile):
            kernel_sum[x, y] = channel[x:x + kernel_shape[0], y:y + kernel_shape[1]].sum()
    return kernel_sum


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

        kernel_sum = compute_kernel_sum(channel, kernel_shape)
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
    grouped = dataframe.groupby("frame")
    tasks = [(t, grouped.get_group(t).reset_index(drop=True).iloc[0], num_vpds, kernel_shape, origin_shape) for t in frame_indices]
    with Pool(cpu_count() // 2) as pool:
        results = list(tqdm.tqdm(
            pool.imap_unordered(process_single_frame_argmax, tasks, chunksize=1),
            total=len(tasks), desc="Processing viewport (legacy)",
            miniters=max(1, len(tasks) // 20)
        ))
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

        kernel_sum = compute_kernel_sum(channel, kernel_shape)
        peaks, _ = get_local_maximums(kernel_sum)
        unique_peaks = get_unique_peaks2(peaks)
        return t, unique_peaks
    except Exception as e:
        print(f"[Worker Error @ frame {t}] {e}")
        traceback.print_exc()
        return t, None


def preprocess_unique_local_maximums_parallel(dataframe: pd.DataFrame, num_vpds: int, kernel_shape, origin_shape, interval: int, get_local_maximums, get_unique_peaks2):
    frame_indices = list(range(0, len(dataframe), interval))
    grouped = dataframe.groupby("frame")
    tasks = [
        (t, grouped.get_group(t).reset_index(drop=True).iloc[0], num_vpds, kernel_shape, origin_shape, get_local_maximums, get_unique_peaks2)
        for t in frame_indices
    ]
    with Pool(cpu_count() // 2) as pool:
        results = list(tqdm.tqdm(
            pool.imap_unordered(process_local_max_frame, tasks, chunksize=1),
            total=len(tasks), desc="Processing viewport (local maximums)",
            miniters=max(1, len(tasks) // 20)
        ))
    results = sorted([r for r in results if r[1] is not None], key=lambda x: x[0])
    return [r[1] for r in results]


def process_all_correct_frame(args):
    t, df_t, num_vpds = args
    arr = np.asarray(df_t.set_index("frame")).squeeze()
    labels = np.split(arr, num_vpds)
    return t, labels


def preprocess_all_correct_parallel(dataframe: pd.DataFrame, num_vpds: int, interval: int):
    grouped = dataframe.groupby("frame")
    frames = [(t, grouped.get_group(t), num_vpds) for t in range(0, len(dataframe), interval)]
    with Pool(cpu_count() // 2) as pool:
        result = list(tqdm.tqdm(
            pool.imap_unordered(process_all_correct_frame, frames, chunksize=1),
            total=len(frames), desc="Processing viewport (all correct)",
            miniters=max(1, len(frames) // 20)
        ))
    result.sort(key=lambda x: x[0])
    return [r[1] for r in result]


def save_all_results(results, path, replay_id=None, coco_dims=None):
    """
    각 프레임마다 별도의 COCO JSON 파일을 생성하도록 변경했습니다.
    - results: list of viewport outputs per frame [(x,y) or (x,y,w,h)]
    - path: 저장 디렉토리
    - replay_id: 파일명에 사용할 리플레이 ID
    - coco_dims: (w, h) 뷰포트 크기, 없으면 config.KERNEL_SHAPE 사용
    """
    import json
    from .Viewport import Viewport

    os.makedirs(path, exist_ok=True)
    
    # 프레임별 JSON 생성
    for t, res in enumerate(results):
        # prepare entries per frame
        entries = []
        # multi-instance case (list of coords)
        if isinstance(res, (list, tuple)) and len(res) > 0 and isinstance(res[0], (list, tuple, np.ndarray)):
            for item in res:
                x, y = int(item[0]), int(item[1])
                if len(item) >= 4:
                    w, h = int(item[2]), int(item[3])
                else:
                    w, h = coco_dims
                entries.append((x, y, w, h))
        else:
            # single-instance
            if len(res) >= 4:
                x, y, w, h = int(res[0]), int(res[1]), int(res[2]), int(res[3])
            else:
                x, y = int(res[0]), int(res[1])
                w, h = coco_dims
            entries.append((x, y, w, h))
        # COCO 구조
        image_entry = {
            "id": 0,
            "file_name": f"input/dst/{replay_id}.rep/{t}.npy",
            "width": int(config.ORIGIN_SHAPE[0]),
            "height": int(config.ORIGIN_SHAPE[1])
        }
        annotations = []
        for ann_id, (x, y, w, h) in enumerate(entries, start=1):
            annotations.append({
                "id": ann_id,
                "image_id": 0,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]],
                "iscrowd": 0
            })

        coco = {
            "info": {"description": f"Viewport frame {t} of replay {replay_id}", "version": "1.0"},
            "licenses": [],
            "images": [image_entry],
            "annotations": annotations,
            "categories": [{"id": 1, "name": "viewport", "supercategory": "viewport"}]
        }

        json_path = os.path.join(path, f"{t}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(coco, f, ensure_ascii=False, indent=2)


def read_single_csv(path):
    try:
        return pd.read_csv(path, index_col=None)
    except Exception as e:
        print(f"[CSV Read Error @ {path}] {e}")
        traceback.print_exc()
        return pd.DataFrame()
