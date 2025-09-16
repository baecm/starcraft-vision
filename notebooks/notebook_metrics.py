import os

import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

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