# src/utils/seed.py
from __future__ import annotations

import os
import random
import numpy as np
import torch
from typing import Optional
from utils.logger import Logger


def set_global_seed(seed: Optional[int], deterministic: bool = True) -> Optional[torch.Generator]:
    """
    Sets global seed across Python, NumPy, PyTorch CPU/CUDA, and configures cuDNN determinism.
    Returns a PyTorch Generator initialized with the given seed.
    """
    if seed is None:
        Logger.info("[Seed] No seed provided; running with default randomness.")
        return None

    seed = int(seed)
    Logger.info(f"[Seed] Setting strict global seed = {seed} (deterministic={deterministic})")

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker initialization function to guarantee multi-worker determinism.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
