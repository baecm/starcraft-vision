import os
import numpy as np
import pandas as pd
import tqdm
from multiprocessing import Process, Queue, cpu_count, Manager
import config


def worker(queue, result_queue, temp_dir, progress_queue):
    while True:
        frame = queue.get()
        if frame is None:
            break
        
        frame_file_npy = os.path.join(temp_dir, f"state_{frame}.npy")
        frame_file_pkl = os.path.join(temp_dir, f"state_{frame}.pkl")
        
        if not os.path.exists(frame_file_npy):
            result_queue.put([])
            progress_queue.put(1)
            continue
        
        frame_data = np.load(frame_file_npy, allow_pickle=True)
        frame_df = pd.DataFrame(frame_data)
        
        frame_df["x_tile"] = frame_df["x"] // config.TILE_SIZE
        frame_df["y_tile"] = frame_df["y"] // config.TILE_SIZE
        frame_df["left_tile"] = frame_df["left"] // config.TILE_SIZE
        frame_df["right_tile"] = frame_df["right"] // config.TILE_SIZE
        frame_df["top_tile"] = frame_df["top"] // config.TILE_SIZE
        frame_df["bottom_tile"] = frame_df["bottom"] // config.TILE_SIZE
        frame_df["width_tile"] = frame_df["right_tile"] - frame_df["left_tile"]
        frame_df["height_tile"] = frame_df["bottom_tile"] - frame_df["top_tile"]
        
        frame_df.loc[frame_df["player"] == "Neutral", ["race", "player_color"]] = ["None", "Cyan"]
        
        os.remove(frame_file_npy)
        frame_df.to_pickle(frame_file_pkl)
        
        result_queue.put(frame_df.to_dict(orient="records"))
        progress_queue.put(1)


class State:
    def __init__(self, data):
        self.data = data
        self.temp_dir = data["temp_dir"]

        self.save_frames()

        self.queue = Queue()
        self.result_queue = Queue()
        self.num_workers = max(1, cpu_count() // 2)
        self.workers = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def save_frames(self):
        grouped = pd.DataFrame(self.data["state_raw"]).groupby("frame")
        for frame, frame_df in grouped:
            np.save(os.path.join(self.temp_dir, f"state_{frame}.npy"), frame_df.to_records(index=False))

    def start_workers(self, progress_queue):
        self.workers = [
            Process(target=worker, args=(self.queue, self.result_queue, self.temp_dir, progress_queue))
            for _ in range(self.num_workers)
        ]
        for p in self.workers:
            p.start()

    def stop_workers(self):
        for _ in self.workers:
            self.queue.put(None)
        for p in self.workers:
            p.join()

    def process(self):
        if not self.data["state_raw"]:
            return

        last_frame = min(self.data["state_raw"][-1]["frame"], self.data["game_length"])
        frames = list(range(0, last_frame + 1))

        with Manager() as manager:
            progress_queue = manager.Queue()
            self.start_workers(progress_queue)

            progress_bar = tqdm.tqdm(
                total=len(frames),
                desc="Processing state...",
                miniters=max(1, len(frames) // 20)
            )

            for frame in frames:
                self.queue.put(frame)
            for _ in frames:
                self.result_queue.get()
                progress_queue.get()
                progress_bar.update(1)
            progress_bar.close()

            self.stop_workers()

        del self.data["state_raw"]

    def run(self):
        self.process()