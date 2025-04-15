import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process, Queue, cpu_count, Manager
from ..starcraft import UnitType, Category


def worker(queue, temp_dir, progress_queue, height, width, players):
    def pivot(df):
        if df.empty:
            return np.zeros((height, width), dtype=int)
        
        result = np.zeros((height, width), dtype=int)
        y_tile, x_tile = df["y_tile"].astype(int).values, df["x_tile"].astype(int).values
        valid_indices = (y_tile < height) & (x_tile < width)
        y_tile, x_tile = y_tile[valid_indices], x_tile[valid_indices]
        
        xy_flat = y_tile * width + x_tile
        counts = np.bincount(xy_flat, minlength=height * width)
        result.flat[:counts.size] = counts
        
        return result.reshape(height, width)
    
    def generate_unit_pivots(df, frame, player_id):
        def get_unit_type(name):
            return UnitType[name] if name in UnitType.__members__ else None
        
        if player_id == "neutral":
            pivot_matrices = np.zeros((1, height, width), dtype=int)
            categories = [Category.RESOURCE]
            
            df["unit_type"] = df["name"].apply(get_unit_type)
            df = df.dropna(subset=["unit_type"])
            
            unit_masks = {
                Category.RESOURCE: df["unit_type"].apply(lambda unit: unit.belongs_to(Category.RESOURCE)),
            }
            
            for idx, category in enumerate(categories):
                if unit_masks[category].any():
                    pivot_matrices[idx] = pivot(df[unit_masks[category]])
            
        else:
            pivot_matrices = np.zeros((4, height, width), dtype=int) if df.empty else np.zeros((4, height, width), dtype=int)
            categories = [Category.WORKER, Category.GROUND, Category.AIR, Category.BUILDING]
            
            df["unit_type"] = df["name"].apply(get_unit_type)
            df = df.dropna(subset=["unit_type"])
            
            unit_masks = {
                Category.WORKER: df["unit_type"].apply(lambda unit: unit.belongs_to(Category.WORKER)),
                Category.GROUND: df["unit_type"].apply(lambda unit: unit.belongs_to(Category.GROUND) and not unit.belongs_to(Category.WORKER, Category.TRIVIAL)),
                Category.AIR: df["unit_type"].apply(lambda unit: unit.belongs_to(Category.AIR) and not unit.belongs_to(Category.WORKER, Category.TRIVIAL)),
                Category.BUILDING: df["unit_type"].apply(lambda unit: unit.belongs_to(Category.BUILDING) and not unit.belongs_to(Category.ADDON)),
            }
            
            for idx, category in enumerate(categories):
                if unit_masks[category].any():
                    pivot_matrices[idx] = pivot(df[unit_masks[category]])
        
        np.save(os.path.join(temp_dir, f"state_{frame}_{player_id}.npy"), pivot_matrices)
    
    while True:
        frame = queue.get()
        if frame is None:
            break
        
        frame_file = os.path.join(temp_dir, f"state_{frame}.pkl")
        if not os.path.exists(frame_file):
            progress_queue.put(1)
            continue
        
        with open(frame_file, "rb") as f:
            frame_df = pd.DataFrame(pickle.load(f))
        
        grouped = frame_df.groupby("player")
        for player, player_df in grouped:
            generate_unit_pivots(player_df, frame, players[player.strip()])
        
        os.remove(os.path.join(temp_dir, f"state_{frame}.pkl"))
        progress_queue.put(1)


class State:
    def __init__(self, data):
        self.data = data
        self.temp_dir = data["temp_dir"]
        self.queue = Queue()
        self.num_workers = max(1, cpu_count() // 2)
        self.workers = []
        
        self.players = {v["name"]: k  for k, v in data["players_data"].items()}
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    def start_workers(self, progress_queue):
        self.workers = [
            Process(target=worker, args=(self.queue, self.temp_dir, progress_queue, self.data["height"], self.data["width"], self.players))
            for _ in range(self.num_workers)
        ]
        for p in self.workers:
            p.start()
    
    def stop_workers(self):
        for _ in self.workers:
            self.queue.put(None)
        for p in self.workers:
            p.join()
    
    def process(self, interval=1):
        frames = list(range(0, self.data["resolution_frame"], interval))
        
        with Manager() as manager:
            progress_queue = manager.Queue()
            self.start_workers(progress_queue)
            
            progress_bar = tqdm(total=len(frames), desc="Converting State...")
            for frame in frames:
                self.queue.put(frame)
            for _ in frames:
                progress_queue.get()
                progress_bar.update(1)
            progress_bar.close()
            
            self.stop_workers()
    
    def run(self, interval=1):
        self.process(interval=interval)
