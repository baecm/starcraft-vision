import os

import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from numpy.lib.stride_tricks import sliding_window_view


def score_density(x: np.ndarray, kw=20, kh=12) -> np.ndarray:
    _, H, W = x.shape
    
    x_sum = x.sum(axis=0, keepdims=True).astype(np.float32, copy=False)  # (N,H,W)
    x_sum = x_sum.reshape(H, W).astype(np.float32, copy=False)  # (H,W)

    ii = np.pad(x_sum, ((1,0),(1,0)), mode='constant')
    ii = ii.cumsum(axis=0).cumsum(axis=1)
    
    win_sum = (
        ii[kh:, kw:]    # bottom-right
        - ii[:-kh, kw:] # top-right
        - ii[kh:, :-kw] # bottom-left
        + ii[:-kh, :-kw]# top-left
    )
    
    den = win_sum / float(kh * kw)                       # (oh,ow)
    # return den
    return (den - den.min()) / (den.max() - den.min())


def gaussian_kernel2d(kw, kh, sigma=None, dtype=np.float32):
    if sigma is None:
        sigma_y = max(kh, 1) / 3.0
        sigma_x = max(kw, 1) / 3.0
    elif np.isscalar(sigma):
        sigma_y = sigma_x = float(sigma)
    else:
        sigma_y, sigma_x = map(float, sigma)

    y = np.arange(kh, dtype=dtype) - (kh - 1) / 2.0
    x = np.arange(kw, dtype=dtype) - (kw - 1) / 2.0
    Y, X = np.meshgrid(y, x, indexing='ij')
    K = np.exp(-(Y**2)/(2*sigma_y**2) - (X**2)/(2*sigma_x**2)).astype(dtype, copy=False)
    K /= (K.sum() + 1e-6)
    return K  # (kh, kw), sum≈1


def score_centeredness(x: np.ndarray, stride=1, sigma=None, kw=20, kh=12) -> np.ndarray:
    _, H, W = x.shape
    
    x_sum = x.sum(axis=0, keepdims=True).astype(np.float32, copy=False)
    x_sum = x_sum.reshape(H, W).astype(np.float32, copy=False)  # (H,W)
    
    K = gaussian_kernel2d(kw, kh, sigma=sigma, dtype=np.float32)  # (kh, kw), sum=1
    denom = float(K.sum()) + 1e-6
    
    patches = sliding_window_view(x_sum, (kh, kw))                     # (N, H-kh+1, W-kw+1, kh, kw)

    cen = (patches * K).sum(axis=(-1, -2))                                  # (N, H-kh+1, W-kw+1)
    
    # 평균 정규화 (PyTorch 경로와 동일)
    denom = float(K.sum()) + 1e-6
    cen = cen / denom

    # 입력 dtype 정책: 부동소수면 입력 dtype으로 반환, 정수 입력이면 float32 유지
    if x.dtype.kind == 'f':
        cen = cen.astype(x.dtype, copy=False)
    # return cen
    return (cen - cen.min()) / (cen.max() - cen.min())
    

def get_projection(x: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    if axis not in (0, 1):
        raise ValueError("axis는 0(높이) 또는 1(너비)이어야 합니다.")
    
    # 축을 따라 합산하여 투영 수행
    projection = x.sum(axis=0 if axis == 0 else 1, keepdims=keepdims)
    
    return projection
    

def score_mixture(x: np.ndarray, stride=1, mode='entropy', fill_when_const=0.0, eps=1e-6, pow_k=1.0, kw=20, kh=12) -> np.ndarray:
    _, H, W = x.shape
    
    # ONES = np.ones((kh, kw), dtype=np.float32)
    xA = x[0:4,...].sum(axis=0, keepdims=False).astype(np.float32, copy=False)  # (H,W)
    xB = x[4:8,...].sum(axis=0, keepdims=False).astype(np.float32, copy=False)  # (H,W)
    
    def _sum_pool2d(x1):
        ii = np.pad(x1, ((1,0),(1,0)), mode='constant')
        ii = ii.cumsum(axis=0).cumsum(axis=1)
        
        win_sum = (
            ii[kh:, kw:]    # bottom-right
            - ii[:-kh, kw:] # top-right
            - ii[kh:, :-kw] # bottom-left
            + ii[:-kh, :-kw]# top-left
        )
        return win_sum
    
    A = _sum_pool2d(xA)  # (oh,ow)
    B = _sum_pool2d(xB)  # (oh,ow)
    
    den = np.maximum(A + B, eps, dtype=np.float32)                       # (oh,ow)
    p = A / den  # (oh,ow)
    
    if mode == 'entropy':
        p1 = np.clip(p, eps, 1.0 - eps)
        q1 = np.clip(1.0 - p, eps, 1.0 - eps)
        
        mix = -(p1 * np.log(p1) + q1 * np.log(q1)) / np.log(2.0)  # (oh,ow), [0,1]
    elif mode == 'gini':
        mix = 2.0 * p * (1.0 - p)  # (oh,ow), [0,0.5]
    elif mode == 'linear':
        mix = 1.0 - np.abs(2.0 * p - 1.0)  # (oh,ow), [0,1]
    else:
        mix = 4.0 * p * (1.0 - p)  # (oh,ow), [0,1]
        
    if pow_k != 1.0:
        mix = np.clip(mix, 0.0, 1.0) ** pow_k  # (oh,ow), [0,1]
        
    # 안전한 min-max 정규화 (NaN/상수 처리)
    mn = np.nanmin(mix)
    mx = np.nanmax(mix)
    rng = mx - mn
    if not np.isfinite(rng) or rng == 0:
        out = np.full_like(mix, fill_when_const, dtype=np.float32)
    else:
        out = (mix - mn) / rng

    # (선택) mix에 NaN이 있었다면 유지하고 싶을 때
    # mask = np.isnan(mix); out[mask] = np.nan

    return out.astype(np.float32, copy=False)


def union_metrics_singlepass(rows, gt_cols, pred_cols,
                             grid_w, grid_h, orig_w, orig_h,
                             kernel_x, kernel_y, use_bbox_size, thresholds,
                             return_summary: bool = False):
    n = len(rows)
    ic_multi = np.full(n, np.nan, dtype=float)
    ic_ratio = np.zeros(n, dtype=float)
    ic_flags = {t: np.zeros(n, dtype=float) for t in thresholds}

    def xywh_to_rect(x,y,w,h):
        if not use_bbox_size:
            w,h = kernel_x, kernel_y
        rx = int(round(x * 32 / orig_w * (grid_w - w)))
        ry = int(round(y * 32 / orig_h * (grid_h - h)))
        rx = max(0, min(grid_w - w, rx))
        ry = max(0, min(grid_h - h, ry))
        return rx, ry, w, h

    for j, (_, row) in enumerate(rows.iterrows()):
        total = np.zeros((grid_h, grid_w), dtype=np.int16)
        have_gt = False
        for gc in gt_cols:
            if gc in row and row[gc] is not None and not (isinstance(row[gc], float) and np.isnan(row[gc])):
                x,y,w,h = (row[gc] if len(row[gc])==4 else tuple(row[gc])[:4])
                rx,ry,rw,rh = xywh_to_rect(x,y,w,h)
                total[ry:ry+rh, rx:rx+rw] += 1
                have_gt = True

        pred_mask = np.zeros((grid_h, grid_w), dtype=bool)
        for pc in pred_cols:
            if pc in row and row[pc] is not None and not (isinstance(row[pc], float) and np.isnan(row[pc])):
                x,y,w,h = (row[pc] if len(row[pc])==4 else tuple(row[pc])[:4])
                rx,ry,rw,rh = xywh_to_rect(x,y,w,h)
                pred_mask[ry:ry+rh, rx:rx+rw] = True

        if not have_gt or not pred_mask.any():
            ic_multi[j] = np.nan
            ic_ratio[j] = 0.0
            for t in thresholds:
                ic_flags[t][j] = 0.0
            continue

        vals = total[pred_mask]
        ic_multi[j] = vals.mean()
        ic_ratio[j] = (vals > 0).mean()
        for t in thresholds:
            ok = (ic_ratio[j] > t) if t == 0.0 else (ic_ratio[j] >= t)
            ic_flags[t][j] = float(ok)

    out = pd.DataFrame({
        ("union","ic_multi"): ic_multi,
        ("union","ic_ratio"): ic_ratio,
        **{("union", f"ic@{int(round(t*100)):03d}"): ic_flags[t] for t in thresholds}
    }, index=rows.index)

    return out


def _union_metrics_worker(args):
    (sub_df, gt_cols, pred_cols, grid_w, grid_h, orig_w, orig_h,
     kernel_x, kernel_y, use_bbox_size, thresholds) = args
    return union_metrics_singlepass(sub_df, gt_cols, pred_cols, grid_w, grid_h,
                                    orig_w, orig_h, kernel_x, kernel_y,
                                    use_bbox_size, thresholds)
    
    
def add_union_metrics_parallel(
    df,
    gt_cols=(("gt","0"),("gt","1"),("gt","2"),("gt","3"),("gt","4")),
    pred_cols=(("pred","0"),),
    grid_w=128, grid_h=128,
    orig_w=3456, orig_h=3720,
    kernel_x=20, kernel_y=12,
    use_bbox_size=False,
    thresholds=(0.0, 0.3, 0.5),
    max_workers=None, chunksize=None
):
    out_cols = [("union","ic_multi"), ("union","ic_ratio")] + \
               [("union", f"ic@{int(round(t*100)):03d}") for t in thresholds]
    for c in out_cols:
        if c not in df.columns:
            df[c] = np.nan

    idx = df.index.to_numpy()
    if chunksize is None:
        nworkers = max_workers or os.cpu_count() or 4
        chunksize = max(1000, len(idx)//(nworkers*4) or 1)
    splits = [idx[i:i+chunksize] for i in range(0, len(idx), chunksize)]

    arglist = [(df.loc[s].copy(), gt_cols, pred_cols, grid_w, grid_h,
                orig_w, orig_h, kernel_x, kernel_y, use_bbox_size, thresholds)
               for s in splits]

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_union_metrics_worker, a) for a in arglist]
        for f in as_completed(futs):
            results.append(f.result())

    out = pd.concat(results).sort_index()
    df.loc[out.index, out.columns] = out.values
    return df