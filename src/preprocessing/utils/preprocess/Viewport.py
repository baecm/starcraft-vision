import os
import glob
import json
from datetime import datetime
import numpy as np
import pandas as pd
import tqdm
from multiprocessing import Pool, cpu_count
from .roci import roci_viewports
from .viewport_parallel_utils import (
    preprocess_argmax_kernel_sum_parallel,
    preprocess_unique_local_maximums_parallel,
    preprocess_all_correct_parallel,
    # save_single_result,
    # save_all_results,
    read_single_csv
)
import traceback
import config


class Viewport:
    def __init__(self, replay_id):
        data_path = os.path.join(os.getcwd(), "data")
        self.viewport_root = os.path.join(data_path, "label", "src")
        self.result_root = os.path.join(data_path, "label", "dst")
        self.replay_id = replay_id
        self.method = None
        self.vpds = []
        self.results = []
                
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def save(self):
        # result_path = os.path.join(self.result_root, self.replay_id + ".rep", self.method)
        # from .viewport_parallel_utils import save_all_results
        # try:
        #     save_all_results(results=self.results, path=result_path, replay_id=self.replay_id, coco_dims=config.KERNEL_SHAPE)
        # except Exception as e:
        #     print("[Save Error]", e)
        #     traceback.print_exc()
        """
        COCO 형식 JSON을 data/label/dst/{replay_id}.rep/{method}/{method}.json
        위치에 하나만 생성합니다.
        """
        # 1) base label 디렉토리 (원래 result_root 는 data/label/src 였으니, dst 로 바꿔줍니다)
        label_base = os.path.join(os.getcwd(), "data", "label", "dst")

        # 2) replay/method 폴더 생성
        out_dir = os.path.join(label_base, f"{self.replay_id}.rep")
        os.makedirs(out_dir, exist_ok=True)

        # 3) 최종 JSON 파일명은 `{method}.json`
        output_path = os.path.join(out_dir, f"{self.method}.json")

        # 4) 결과 내보내기
        try:
            self.export_to_coco(
                output_path=output_path,
                viewport_dims=config.KERNEL_SHAPE
            )
            print(f"[Save] COCO JSON saved to {output_path}")
        except Exception as e:
            print("[Save Error]", e)
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
                df = (
                    df.set_index("frame")
                      .reindex(range(df["frame"].max()))
                      .ffill()
                      .reset_index()
                      .astype(int)
                      .set_index("frame")
                )
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
            df = (df / config.TILE_SIZE).astype(int).reset_index()
            return df, num
        except Exception as e:
            print("[Merge Error]", e)
            traceback.print_exc()
            return pd.DataFrame(), 0

    def preprocess_argmax_kernel_sum(self, dataframe, num_vpds):
        return preprocess_argmax_kernel_sum_parallel(
            dataframe, num_vpds, config.KERNEL_SHAPE, config.ORIGIN_SHAPE, 1
        )

    def preprocess_unique_local_maximums(self, dataframe, num_vpds):
        from local_peaks import get_local_maximums, get_unique_peaks2
        return preprocess_unique_local_maximums_parallel(
            dataframe, num_vpds, config.KERNEL_SHAPE, config.ORIGIN_SHAPE, 1,
            get_local_maximums, get_unique_peaks2
        )

    def preprocess_all_correct(self, dataframe, num_vpds):
        return preprocess_all_correct_parallel(
            dataframe, num_vpds, 1
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

        total_frames = range(0, len(dataframe), 1)
        for t in tqdm.tqdm(
            total_frames,
            desc="Processing viewport(consider previous)",
            miniters=max(1, len(total_frames) // 20)
        ):
            try:
                df_t = dataframe.loc[dataframe["frame"] == t].squeeze()
                channel = np.zeros(config.ORIGIN_SHAPE)
                kernel = np.ones(config.KERNEL_SHAPE)
                for i in range(num_vpds):
                    x = int(df_t[f"vpx_{i + 1}"])
                    y = int(df_t[f"vpy_{i + 1}"])
                    channel[x:x + config.KERNEL_SHAPE[0], y:y + config.KERNEL_SHAPE[1]] += kernel
                channel = channel.T

                width_tile = config.ORIGIN_SHAPE[0] - config.KERNEL_SHAPE[0]
                height_tile = config.ORIGIN_SHAPE[1] - config.KERNEL_SHAPE[1]
                kernel_sum = np.zeros((width_tile, height_tile))
                for x in range(width_tile):
                    for y in range(height_tile):
                        kernel_sum[x][y] = channel[x:x + config.KERNEL_SHAPE[0], y:y + config.KERNEL_SHAPE[1]].sum()

                peaks, _ = get_local_maximums(kernel_sum)
                unique_peaks = get_unique_peaks(peaks)

                current = compare_with_previous(kernel_sum, unique_peaks, previous_viewport)
                previous_viewport = current
                result.append(current)
            except Exception as e:
                print(f"[Error at frame {t}]", e)
                traceback.print_exc()
        return result

    def apply_roci(self, results):
        """Append the common-interest viewports to each frame's labels.

        Joo et al. train on "5 + n" targets per frame: the five observer
        viewports, plus however many regions of common interest that frame
        turned out to have. The count varies by frame, which is why this
        appends rather than pads.
        """
        height, width = config.ORIGIN_SHAPE
        box_wh = config.KERNEL_SHAPE
        added = 0
        augmented = []
        for res in results:
            if isinstance(res, (list, tuple)):
                observers = [np.asarray(r).flatten() for r in res]
            else:
                observers = [np.asarray(res).flatten()]

            points = [(int(a[0]), int(a[1])) for a in observers if a.size >= 2]
            extra = roci_viewports(
                points, height, width, box_wh,
                sigma=config.ROCI_SIGMA,
                min_sep=config.ROCI_MIN_SEP,
                threshold=config.ROCI_THRESHOLD,
                max_regions=config.ROCI_MAX_REGIONS,
            )
            added += len(extra)
            augmented.append(list(observers) + [np.array(p) for p in extra])

        per_frame = added / max(1, len(results))
        print(f"[Viewport] ROCI added {added} viewports "
              f"({per_frame:.2f} per frame)")
        return augmented

    def run(self, method, roci=None):
        if roci is None:
            roci = config.ROCI_ENABLED
        print(f"[Viewport] Method selected: {method} (roci={roci})")
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

            # ROCI labels are written beside the plain ones rather than over
            # them: the two are different training sets, and a run has to be
            # able to say which it used.
            self.method = f"{method}_roci" if roci else method
            print(f"[Viewport] Running preprocessing: {method}")
            self.results = methods[method](dataframe, num_vpds)
            if roci:
                print("[Viewport] Applying ROCI target augmentation...")
                self.results = self.apply_roci(self.results)
            print(f"[Viewport] Finished preprocessing with method: {self.method}")
        except Exception as e:
            print("[Viewport.run] Exception occurred")
            traceback.print_exc()

    def export_to_coco(self, output_path, viewport_dims=None):
        """
        Export viewport results to COCO-style JSON.
        - self.results: list of (x, y) or (x, y, w, h).
        - viewport_dims: (w, h) applied if results only provide x, y.
        """
        # COCO skeleton
        coco = {
            "info": {
                "description": f"Viewport annotations for replay {self.replay_id}",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "viewport", "supercategory": "viewport"}
            ]
        }
        # Image dims
        H, W = config.ORIGIN_SHAPE
        # Images entries
            
        for fid in range(len(self.results)):
            input_dir = os.path.join(os.getcwd(), "data", "input", "dst", self.replay_id + ".rep")
            
            if not os.path.isfile(os.path.join(input_dir, f"{fid}.npy")):
                continue
            
            coco["images"].append({
                "id": int(fid),
                "file_name": f"input/dst/{self.replay_id}.rep/{fid}.npy",
                "width": int(W),
                "height": int(H)
            })
        # Annotations
        ann_id = 1
        for fid, res in enumerate(self.results):
            input_dir = os.path.join(os.getcwd(), "data", "input", "dst", self.replay_id + ".rep")
            
            if not os.path.isfile(os.path.join(input_dir, f"{fid}.npy")):
                continue
            
            if isinstance(res, (list, tuple)):
                coords = [np.asarray(r).flatten() for r in res]
            else:
                coords = [np.asarray(res).flatten()]

            for arr in coords:
                if arr.size == 2:
                    x, y = int(arr[0]), int(arr[1])
                    w, h = viewport_dims or config.KERNEL_SHAPE
                elif arr.size >= 4:
                    x, y, w, h = map(int, arr[:4])
                else:
                    raise ValueError(f"Unexpected res shape {arr.shape}")

                coco["annotations"].append({
                    "id":          ann_id,
                    "image_id":    fid,
                    "category_id": 1,
                    "bbox":        [x, y, w, h],
                    "area":        int(w * h),
                    "segmentation":[[x, y, x+w, y, x+w, y+h, x, y+h]],
                    "iscrowd":     0
                })
                ann_id += 1

        # 3) JSON 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)
