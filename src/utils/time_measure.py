from contextlib import contextmanager

import time
import torch


@contextmanager
def measure_time(task_name: str, sync_cuda: bool = True):
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    box = {}
    yield box
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    box["elapsed"] = end - start
    print(f"[{task_name}] {box['elapsed']:.6f} sec", flush=True)