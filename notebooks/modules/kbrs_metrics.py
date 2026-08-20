import os
import sys
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# 앞서 분리한 상수 모듈 임포트
from modules.constants import KW, KH, GT_COLS


# =====================================================================
# 1. KBRS 스코어 계산 로직 (유지) - 시각화 및 BRS 맵 생성용
# =====================================================================

def score_density(x: np.ndarray) -> np.ndarray:
    """유닛 및 건물의 밀도(Density)를 계산하여 스코어 맵으로 반환합니다."""
    _, H, W = x.shape
    x_sum = x.sum(axis=0, keepdims=True).astype(np.float32, copy=False)
    x_sum = x_sum.reshape(H, W).astype(np.float32, copy=False)

    ii = np.pad(x_sum, ((1,0),(1,0)), mode='constant')
    ii = ii.cumsum(axis=0).cumsum(axis=1)
    
    win_sum = (
        ii[KH:, KW:] - ii[:-KH, KW:] - ii[KH:, :-KW] + ii[:-KH, :-KW]
    )
    
    den = win_sum / float(KH * KW)
    return (den - den.min()) / (den.max() - den.min())


def gaussian_kernel2d(kw, kh, sigma=None, dtype=np.float32):
    """2D 가우시안 커널을 생성합니다."""
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
    return K


def score_centeredness(x: np.ndarray, stride=1, sigma=None) -> np.ndarray:
    """가우시안 커널을 이용해 중심성(Centeredness) 스코어를 계산합니다."""
    _, H, W = x.shape
    x_sum = x.sum(axis=0, keepdims=True).astype(np.float32, copy=False)
    x_sum = x_sum.reshape(H, W).astype(np.float32, copy=False)
    
    K = gaussian_kernel2d(KW, KH, sigma=sigma, dtype=np.float32)
    
    patches = sliding_window_view(x_sum, (KH, KW))
    cen = (patches * K).sum(axis=(-1, -2))
    
    denom = float(K.sum()) + 1e-6
    cen = cen / denom

    if x.dtype.kind == 'f':
        cen = cen.astype(x.dtype, copy=False)
        
    return (cen - cen.min()) / (cen.max() - cen.min())


def score_mixture(x: np.ndarray, stride=1, mode='entropy', fill_when_const=0.0) -> np.ndarray:
    """두 플레이어 간의 교전/혼합(Mixture) 정도를 스코어로 계산합니다."""
    _, H, W = x.shape
    eps = 1e-6
    pow_k = 1.0
    
    xA = x[0:4,...].sum(axis=0, keepdims=False).astype(np.float32, copy=False)
    xB = x[4:8,...].sum(axis=0, keepdims=False).astype(np.float32, copy=False)
    
    def _sum_pool2d(x1):
        ii = np.pad(x1, ((1,0),(1,0)), mode='constant')
        ii = ii.cumsum(axis=0).cumsum(axis=1)
        win_sum = (ii[KH:, KW:] - ii[:-KH, KW:] - ii[KH:, :-KW] + ii[:-KH, :-KW])
        return win_sum
    
    A = _sum_pool2d(xA)
    B = _sum_pool2d(xB)
    
    den = np.maximum(A + B, eps, dtype=np.float32)
    p = A / den
    
    if mode == 'entropy':
        p1 = np.clip(p, eps, 1.0 - eps)
        q1 = np.clip(1.0 - p, eps, 1.0 - eps)
        mix = -(p1 * np.log(p1) + q1 * np.log(q1)) / np.log(2.0)
    elif mode == 'gini':
        mix = 2.0 * p * (1.0 - p)
    elif mode == 'linear':
        mix = 1.0 - np.abs(2.0 * p - 1.0)
    else:
        mix = 4.0 * p * (1.0 - p)
        
    if pow_k != 1.0:
        mix = np.clip(mix, 0.0, 1.0) ** pow_k
        
    mn = np.nanmin(mix)
    mx = np.nanmax(mix)
    rng = mx - mn
    
    if not np.isfinite(rng) or rng == 0:
        out = np.full_like(mix, fill_when_const, dtype=np.float32)
    else:
        out = (mix - mn) / rng

    return out.astype(np.float32, copy=False)


def get_projection(x: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    if axis not in (0, 1):
        raise ValueError("axis는 0(높이) 또는 1(너비)이어야 합니다.")
    return x.sum(axis=0 if axis == 0 else 1, keepdims=keepdims)


def _worker_compute(input_dst_dir, replay, image_id, weights):
    """워커 프로세스에서 스코어를 사전 계산합니다."""
    path = os.path.join(input_dst_dir, str(replay), f"{image_id}.npy")
    arr  = np.load(path, mmap_mode="r")
    
    den  = score_density(arr).astype(np.float32, copy=False)
    cen  = score_centeredness(arr).astype(np.float32, copy=False)
    mix  = score_mixture(arr, mode="linear").astype(np.float32, copy=False)

    sw   = weights['density'] + weights['mixture'] + weights['centeredness']
    sw   = sw if sw > 1e-12 else 1.0
    brs  = (weights['density'] * den + weights['mixture'] * mix + weights['centeredness'] * cen) / sw
    brs  = np.clip(brs, 0.0, 1.0, out=brs)

    return {"path": path, "den": den, "cen": cen, "mix": mix, "brs": brs}


def _extract_gts(item_row):
    """Pandas Series(row)에서 GT(Ground Truth) 좌표 정보를 안전하게 추출합니다."""
    gts = []
    for k in GT_COLS:
        if k in item_row:
            v = item_row[k]
            if v is None: 
                continue
            arr = np.asarray(v, dtype=float).ravel()
            if arr.size == 4 and not np.any(np.isnan(arr)):
                x, y, w, h = arr
                gts.append((x, y, w, h))
    return gts


# =====================================================================
# 2. 공식 IC 평가 로직 (추가됨) - custom_evaluator 연동
# =====================================================================

# 외부 모듈(custom_evaluator) 임포트를 위한 sys.path 설정
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../project/modules
project_root = os.path.dirname(current_dir)               # .../project
src_dir = os.path.abspath(os.path.join(project_root, "../src"))

if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from metrics.custom_evaluator import eval_intersection_run
except ImportError:
    from metrics import eval_intersection_run


def build_agent_traces_from_df(gt_df, pred_df, width=128, height=128, max_x=3456.0, max_y=3720.0):
    """Pandas DataFrame을 evaluate.py에서 사용하는 labels_tests 포맷으로 변환합니다."""
    if pred_df is None or pred_df.empty or 'image_id' not in pred_df.columns:
        return [], []

    # 1. Prediction은 최고 점수(Top-1) 1개만 사용
    pred_top1 = pred_df.sort_values(by=['image_id', 'score'], ascending=[True, False]).groupby('image_id').head(1)
    
    # 2. image_id별로 박스 리스트 그룹화
    gt_grouped = gt_df.groupby('image_id')['bbox'].apply(list).to_dict()
    pred_grouped = pred_top1.groupby('image_id')['bbox'].apply(list).to_dict()
    
    labels_tests = []
    image_ids = []
    
    for iid, gts in gt_grouped.items():
        preds = pred_grouped.get(iid, [])
        
        if len(preds) == 0:
            # 예측 실패 프레임: 화면 밖 더미 좌표 부여 -> IC = 0.0 페널티 공정 반영
            dummy_vx, dummy_vy = -9999.0, -9999.0
            agent0 = [{"vpx": dummy_vx, "vpy": dummy_vy}]
        else:
            px, py = preds[0][:2]
            pcx = px + KW / 2.0
            pcy = py + KH / 2.0
            
            vx = float(pcx) / max(1, (width - KW)) * max_x
            vy = float(pcy) / max(1, (height - KH)) * max_y
            agent0 = [{"vpx": vx, "vpy": vy}]
            
        # GT 에이전트 생성
        ref_agents = []
        for g_box in gts:
            gx, gy = g_box[:2]
            gcx = gx + KW / 2.0
            gcy = gy + KH / 2.0
            
            gvx = float(gcx) / max(1, (width - KW)) * max_x
            gvy = float(gcy) / max(1, (height - KH)) * max_y
            ref_agents.append([{"vpx": gvx, "vpy": gvy}])
            
        labels_tests.append([agent0] + ref_agents)
        image_ids.append(iid)
        
    return labels_tests, image_ids


def get_official_metrics_df(gt_df, pred_df):
    """custom_evaluator를 실행하여 공식 IC 지표가 포함된 DataFrame을 반환합니다."""
    labels_tests, image_ids = build_agent_traces_from_df(gt_df, pred_df)
    
    if not labels_tests:
        return pd.DataFrame()
        
    per_image, agg = eval_intersection_run(
        labels_tests,
        x_len=KW, y_len=KH,
        width=128, height=128,
        max_x=3456.0, max_y=3720.0
    )
    
    results = []
    for iid, img_ir in zip(image_ids, per_image):
        ir = float(img_ir.ir)
        results.append({
            'image_id': iid,
            'ir': ir,
            'ic@000': ir > 0.0,
            'ic@030': ir >= 0.30,
            'ic@050': ir >= 0.50
        })
        
    return pd.DataFrame(results)