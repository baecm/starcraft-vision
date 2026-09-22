# src/utils/torch_compat.py
"""Runtime guards for torch features the container cannot support."""
from __future__ import annotations

import os

import torch

from utils.logger import Logger


def disable_inductor() -> None:
    """Stop torch.compile from being attempted, and never let it be fatal.

    Recent torchvision wraps `ops.roi_align` in a torch.compile hook. Inductor
    then reaches Triton, which shells out to a C compiler to build its CUDA
    utils module - and the pytorch runtime image has no gcc, so Mask R-CNN
    inference dies with "Failed to find C compiler" the first time roi_align is
    called. Nothing here needs compiled kernels, so the compile is disabled
    rather than made to work.

    Both the env-var and the config route are used because the variable name
    has changed across torch versions and the workstations are not all on the
    same image - one reports a traceback through _codegen_partitions, another
    through an older scheduler path. suppress_errors is the backstop: if a
    compile is attempted anyway, it falls back to eager instead of raising.
    """
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    try:
        import torch._dynamo as dynamo

        dynamo.config.suppress_errors = True
        if hasattr(dynamo.config, "disable"):
            dynamo.config.disable = True
        Logger.info("[TorchCompat] torch.compile disabled; inductor errors fall back to eager.")
    except Exception as e:  # pragma: no cover - depends on the torch build
        Logger.warn(f"[TorchCompat] Could not configure torch._dynamo: {type(e).__name__}: {e}")
