# src/model/kbrs_kernel.py
import torch

def make_ones_kernel(kh: int, kw: int) -> torch.Tensor:
    return torch.ones((1, 1, kh, kw), dtype=torch.float32)

def make_center_kernel(kh: int, kw: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(kh, dtype=torch.float32),
        torch.arange(kw, dtype=torch.float32),
        indexing="ij",
    )
    cy, cx = (kh - 1) / 2.0, (kw - 1) / 2.0
    sigma = kh / 4.0
    w = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (sigma ** 2)))
    return w.view(1, 1, kh, kw)
