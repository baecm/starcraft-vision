#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KBRS cache/lookup CLI (subcommands)

A안: 서브커맨드로 분리
  - cache : feature(.npy)마다 KBRS grid cache(.npz) 생성/갱신
  - lookup: GT 또는 pred bbox 중심 좌표에 대해 cache에서 KBRS 값을 lookup하여 CSV 출력

핵심 아이디어:
- KBRS grid cache는 feature만으로 deterministic하게 계산 가능(= GT/model과 무관)
- lookup 단계에서만 GT/pred에 따라 (x,y) 좌표 집합이 달라진다.

주의(경계 처리):
- cache는 기본적으로 "valid grid"만 정의한다.
- lookup은 기본적으로 (x,y)를 grid 범위로 clamp하여 캐시와 일관되게 값을 뽑는다(--border clamp).
- 기존 on-the-fly(경계에서 잘린 패치 허용)와 1:1 비교가 필요하면 --border on_the_fly.

사용 예:
  # 1) 캐시 생성(리플레이 2개, 병렬 16)
  python kbrs_cli.py cache --replays 275 3613 --num-workers 16

  # 2) GT 중심 lookup CSV
  python kbrs_cli.py lookup --source gt --replays 275 --csv-out /tmp/kbrs_gt_275.csv --num-workers 8

  # 3) model pred 중심 lookup CSV
  python kbrs_cli.py lookup --source pred --replays 275 --model-name XXX --epoch 30 --csv-out /tmp/kbrs_pred.csv
"""

from __future__ import annotations

import os
import json
import argparse
from typing import List, Dict, Tuple, Optional, Any, Iterable

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from multiprocessing import Pool, cpu_count


# ---------------------------------------------------------------------
# KBRS core
# ---------------------------------------------------------------------
def make_gaussian_kernel(h: int, w: int, sigma: Optional[float] = None) -> np.ndarray:
    """
    (h, w) 크기의 2D 가우시안 커널을 생성한다.

    Args:
        h: 커널 높이
        w: 커널 너비
        sigma: 표준편차. None이면 h/4.0을 사용한다.

    Returns:
        shape=(h, w) float32 가우시안 커널.

    Notes:
        centeredness 계산에서 binary 맵에 가중치로 곱해 사용한다.
    """
    if sigma is None:
        sigma = h / 4.0
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    dist2 = (ys - cy) ** 2 + (xs - cx) ** 2
    g = np.exp(-dist2 / (2.0 * sigma * sigma))
    return g.astype(np.float32)


def kbrs_scores_from_patch(patch: np.ndarray, threshold: float = 0.0) -> Tuple[float, float, float]:
    """
    단일 패치에서 KBRS 스코어(density/centeredness/mixture)를 계산한다.

    Args:
        patch: (C,H,W) 또는 (H,W)
        threshold: binary map 기준. patch2d > threshold 를 1로 간주한다.

    Returns:
        (density, centeredness, mixture)
          - density: binary 평균(= 1 비율) [0,1]
          - centeredness: gaussian weighted mean [0,1]
          - mixture: density * centeredness

    Notes:
        (C,H,W)이면 채널 max projection으로 2D 맵을 만든다.
    """
    if patch.ndim == 3:
        patch2d = patch.max(axis=0)
    elif patch.ndim == 2:
        patch2d = patch
    else:
        raise ValueError(f"Unsupported patch shape: {patch.shape}")

    if patch2d.size == 0:
        return 0.0, 0.0, 0.0

    binary = (patch2d > threshold).astype(np.float32)
    density = float(binary.mean())

    h, w = binary.shape
    g = make_gaussian_kernel(h, w, sigma=h / 4.0)
    g_sum = float(g.sum()) + 1e-12

    centered_raw = float((binary * g).sum() / g_sum)
    centeredness = float(max(0.0, min(1.0, centered_raw)))
    mixture = density * centeredness
    return density, centeredness, mixture


def extract_window(feat: np.ndarray, center_x: float, center_y: float, win_w: int, win_h: int) -> np.ndarray:
    """
    feature 맵에서 (center_x, center_y)를 중심으로 win_w x win_h 패치를 잘라낸다.

    Args:
        feat: (C,H,W) 또는 (H,W)
        center_x: 중심 x(픽셀)
        center_y: 중심 y(픽셀)
        win_w: 윈도우 너비
        win_h: 윈도우 높이

    Returns:
        경계에서 잘린 "부분 패치"를 포함할 수 있는 패치 배열.
        유효 영역이 없으면 빈 배열(0:0 슬라이스).

    Notes:
        --border on_the_fly에서 기존 방식 재현용으로 사용한다.
    """
    if feat.ndim == 3:
        _, H, W = feat.shape
    elif feat.ndim == 2:
        H, W = feat.shape
    else:
        raise ValueError(f"Unsupported feature shape: {feat.shape}")

    half_w = win_w // 2
    half_h = win_h // 2

    x0 = int(round(center_x)) - half_w
    y0 = int(round(center_y)) - half_h
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W, x0 + win_w)
    y1 = min(H, y0 + win_h)

    if x1 <= x0 or y1 <= y0:
        if feat.ndim == 3:
            return feat[:, 0:0, 0:0]
        return feat[0:0, 0:0]

    if feat.ndim == 3:
        return feat[:, y0:y1, x0:x1]
    return feat[y0:y1, x0:x1]


def compute_kbrs_grid(
    feat: np.ndarray,
    win_w: int,
    win_h: int,
    stride_x: int,
    stride_y: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    feature 전체를 valid 슬라이딩 윈도우로 스캔해 KBRS map 3종을 만든다.

    Args:
        feat: (C,H,W) 또는 (H,W)
        win_w, win_h: 윈도우 크기
        stride_x, stride_y: stride
        threshold: binary map threshold

    Returns:
        (density_map, centeredness_map, mixture_map)
        각 map은 shape=(out_h, out_w) (y,x 순서)

    Notes:
        out_h = (H - win_h)//stride_y + 1
        out_w = (W - win_w)//stride_x + 1
        즉 윈도우가 "완전히 들어가는(valid)" 위치만 계산한다.
    """
    if feat.ndim == 3:
        _, H, W = feat.shape
    elif feat.ndim == 2:
        H, W = feat.shape
    else:
        raise ValueError(f"Unsupported feature shape: {feat.shape}")

    out_h = max(0, (H - win_h) // stride_y + 1)
    out_w = max(0, (W - win_w) // stride_x + 1)

    density_map = np.zeros((out_h, out_w), dtype=np.float32)
    centeredness_map = np.zeros((out_h, out_w), dtype=np.float32)
    mixture_map = np.zeros((out_h, out_w), dtype=np.float32)

    for oy in range(out_h):
        for ox in range(out_w):
            cx = ox * stride_x + win_w / 2.0
            cy = oy * stride_y + win_h / 2.0
            patch = extract_window(feat, cx, cy, win_w, win_h)
            d, c, m = kbrs_scores_from_patch(patch, threshold=threshold)
            density_map[oy, ox] = d
            centeredness_map[oy, ox] = c
            mixture_map[oy, ox] = m

    return density_map, centeredness_map, mixture_map


# ---------------------------------------------------------------------
# COCO + path utils
# ---------------------------------------------------------------------
def centroid_from_coco_ann(ann: dict, img_w: int, img_h: int) -> Tuple[float, float]:
    """
    COCO annotation(dict)에서 bbox 중심 좌표를 계산한다.

    Args:
        ann: COCO annotation 또는 detection dict. 'bbox'가 [x,y,w,h]로 있다고 가정한다.
        img_w, img_h: bbox가 없을 때 fallback(이미지 중앙)에 사용.

    Returns:
        (cx, cy): bbox 중심(픽셀 float)
    """
    if "bbox" in ann and ann["bbox"] is not None:
        x, y, w, h = ann["bbox"]
        return float(x + w / 2.0), float(y + h / 2.0)
    return float(img_w) / 2.0, float(img_h) / 2.0


def load_coco_gt(label_root: str, replay_id: str, label_method: str) -> COCO:
    """
    GT COCO JSON을 로드한다.

    Path 규칙:
        {label_root}/{replay_id}.rep/{label_method}.json

    Args:
        label_root: GT 라벨 루트
        replay_id: 리플레이 ID
        label_method: 라벨 파일 베이스명

    Returns:
        pycocotools.COCO 객체

    Raises:
        FileNotFoundError: 파일이 없을 때
    """
    gt_dir = os.path.join(label_root, f"{replay_id}.rep")
    gt_path = os.path.join(gt_dir, f"{label_method}.json")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Ground truth (COCO) file not found: {gt_path}")
    return COCO(gt_path)


def load_coco_preds(pred_root: str, model_name: str, epoch: int, replay_id: str, label_method: str) -> Dict[int, List[dict]]:
    """
    모델 prediction COCO JSON을 로드하고 image_id별로 grouping하여 반환한다.

    Path 규칙:
        {pred_root}/{model_name}/model_{epoch}/{replay_id}.rep/{label_method}.json

    Args:
        pred_root: prediction 루트
        model_name: 모델 이름
        epoch: epoch 번호
        replay_id: 리플레이 ID
        label_method: 라벨 파일 베이스명

    Returns:
        preds_by_img: Dict[image_id -> list of detection dict]

    Raises:
        FileNotFoundError: 파일이 없을 때
        ValueError: JSON 포맷이 예상과 다를 때
    """
    pred_dir = os.path.join(pred_root, model_name, f"model_{epoch}", f"{replay_id}.rep")
    pred_path = os.path.join(pred_dir, f"{label_method}.json")
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    with open(pred_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    if isinstance(loaded, dict) and "annotations" in loaded:
        dets = loaded["annotations"]
    elif isinstance(loaded, list):
        dets = loaded
    else:
        raise ValueError("Unsupported prediction JSON format for COCO dets.")

    preds_by_img: Dict[int, List[dict]] = {}
    for det in dets:
        img_id = int(det["image_id"])
        preds_by_img.setdefault(img_id, []).append(det)
    return preds_by_img


def build_feature_path_from_meta(
    input_root: str,
    replay_id: str,
    file_name: str,
    img_id: int,
    use_file_name: bool,
    feature_ext: str,
) -> str:
    """
    COCO image 메타정보로 feature(.npy) 경로를 만든다.

    Path 규칙:
        {input_root}/{replay_id}.rep/{file_name_base or img_id}{feature_ext}

    Args:
        input_root: feature 루트
        replay_id: 리플레이 ID
        file_name: COCO image file_name
        img_id: COCO image id
        use_file_name: True면 file_name 기반, False면 img_id 기반
        feature_ext: 확장자(기본 .npy)

    Returns:
        feature 파일 경로
    """
    rep_dir = os.path.join(input_root, f"{replay_id}.rep")
    if use_file_name and file_name:
        base, _ext = os.path.splitext(file_name)
        fname = base + feature_ext
    else:
        fname = f"{img_id}{feature_ext}"
    return os.path.join(rep_dir, fname)


# ---------------------------------------------------------------------
# Cache utils
# ---------------------------------------------------------------------
def default_cache_root(input_root: str) -> str:
    """
    기본 캐시 루트 디렉터리 경로를 반환한다.

    Args:
        input_root: feature 루트

    Returns:
        {input_root}/__kbrs_cache__
    """
    return os.path.join(input_root, "__kbrs_cache__")


def cache_tag(win_w: int, win_h: int, sx: int, sy: int, thr: float) -> str:
    """
    캐시 파일명에 포함할 파라미터 tag를 만든다.

    Args:
        win_w, win_h: 윈도우 크기
        sx, sy: stride
        thr: threshold

    Returns:
        예) "w20h12_sx1sy1_thr0p0"
    """
    thr_s = str(thr).replace(".", "p").replace("-", "m")
    return f"w{win_w}h{win_h}_sx{sx}sy{sy}_thr{thr_s}"


def build_cache_path(
    cache_root: str,
    replay_id: str,
    feature_path: str,
    win_w: int,
    win_h: int,
    sx: int,
    sy: int,
    thr: float,
) -> str:
    """
    feature 파일에 대응하는 캐시(.npz) 경로를 생성한다.

    Path 규칙:
        {cache_root}/{replay_id}.rep/{feature_base}.kbrs.{tag}.npz

    Args:
        cache_root: 캐시 루트
        replay_id: 리플레이 ID
        feature_path: 원본 feature 파일 경로
        win_w, win_h, sx, sy, thr: 캐시 파라미터

    Returns:
        캐시 파일 경로
    """
    rep_dir = os.path.join(cache_root, f"{replay_id}.rep")
    os.makedirs(rep_dir, exist_ok=True)

    base = os.path.basename(feature_path)
    base_noext, _ = os.path.splitext(base)
    tag = cache_tag(win_w, win_h, sx, sy, thr)
    return os.path.join(rep_dir, f"{base_noext}.kbrs.{tag}.npz")


def save_kbrs_cache(
    cache_path: str,
    density_map: np.ndarray,
    centeredness_map: np.ndarray,
    mixture_map: np.ndarray,
    meta: Dict[str, Any],
    dtype: str,
) -> None:
    """
    KBRS map 3종과 메타데이터를 .npz로 저장한다.

    Args:
        cache_path: 저장 경로
        density_map, centeredness_map, mixture_map: KBRS map들(shape=(out_h,out_w))
        meta: 파라미터/shape/원본 경로 등 메타데이터
        dtype: "float16" 또는 "float32"

    Notes:
        - meta는 JSON 문자열(meta_json)로 직렬화해 저장한다.
        - 저장은 np.savez_compressed 사용.
    """
    if dtype == "float16":
        d = density_map.astype(np.float16, copy=False)
        c = centeredness_map.astype(np.float16, copy=False)
        m = mixture_map.astype(np.float16, copy=False)
    elif dtype == "float32":
        d = density_map.astype(np.float32, copy=False)
        c = centeredness_map.astype(np.float32, copy=False)
        m = mixture_map.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported cache dtype: {dtype}")

    meta_json = json.dumps(meta, ensure_ascii=False)
    np.savez_compressed(
        cache_path,
        density=d,
        centeredness=c,
        mixture=m,
        meta_json=np.array([meta_json], dtype=object),
    )


def load_kbrs_cache(cache_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    KBRS cache(.npz)를 로드한다.

    Args:
        cache_path: 캐시 파일 경로

    Returns:
        (density, centeredness, mixture, meta)
          - map은 float32로 변환하여 반환
          - meta는 dict로 복원
    """
    with np.load(cache_path, allow_pickle=True) as z:
        density = z["density"].astype(np.float32)
        centeredness = z["centeredness"].astype(np.float32)
        mixture = z["mixture"].astype(np.float32)
        meta_json = str(z["meta_json"][0])
        meta = json.loads(meta_json)
    return density, centeredness, mixture, meta


def ensure_kbrs_cache(
    replay_id: str,
    feature_path: str,
    cache_root: str,
    win_w: int,
    win_h: int,
    sx: int,
    sy: int,
    thr: float,
    cache_dtype: str,
    rebuild: bool,
) -> Optional[str]:
    """
    캐시가 있으면 재사용하고, 없으면 생성한 뒤 캐시 경로를 반환한다.

    Args:
        replay_id: 리플레이 ID
        feature_path: 원본 feature(.npy) 경로
        cache_root: 캐시 루트
        win_w, win_h, sx, sy, thr: 캐시 파라미터
        cache_dtype: 캐시 저장 dtype
        rebuild: True면 캐시가 있어도 강제 재생성

    Returns:
        캐시 파일 경로(성공) 또는 feature가 없으면 None

    Notes:
        - 파라미터(tag)가 파일명에 포함되므로, 파라미터가 다르면 다른 캐시가 생성된다.
        - 여기서는 meta 검증을 최소화하고(tag가 사실상 구분), rebuild로 강제 갱신을 지원한다.
    """
    if not os.path.isfile(feature_path):
        return None

    cpath = build_cache_path(cache_root, replay_id, feature_path, win_w, win_h, sx, sy, thr)
    if os.path.isfile(cpath) and not rebuild:
        return cpath

    feat = np.load(feature_path)
    density, centeredness, mixture = compute_kbrs_grid(feat, win_w, win_h, sx, sy, thr)

    if feat.ndim == 3:
        _, H, W = feat.shape
    else:
        H, W = feat.shape

    meta = {
        "feature_path": feature_path,
        "feature_shape": list(feat.shape),
        "H": int(H),
        "W": int(W),
        "win_w": int(win_w),
        "win_h": int(win_h),
        "stride_x": int(sx),
        "stride_y": int(sy),
        "threshold": float(thr),
        "grid_h": int(density.shape[0]),
        "grid_w": int(density.shape[1]),
        "grid_mode": "valid",
    }
    save_kbrs_cache(cpath, density, centeredness, mixture, meta, dtype=cache_dtype)
    return cpath


def xy_to_grid_index(
    x: float,
    y: float,
    win_w: int,
    win_h: int,
    sx: int,
    sy: int,
    grid_w: int,
    grid_h: int,
    border: str,
) -> Optional[Tuple[int, int]]:
    """
    (x,y) 픽셀 좌표를 KBRS grid 인덱스(oy, ox)로 매핑한다.

    Args:
        x, y: 픽셀 좌표
        win_w, win_h: 윈도우 크기
        sx, sy: stride
        grid_w, grid_h: KBRS map 크기(out_w,out_h)
        border:
            - "clamp": 범위를 벗어나면 가장 가까운 셀로 clamp
            - "skip": 범위를 벗어나면 None 반환

    Returns:
        (oy, ox) 또는 None

    Notes:
        cache grid의 중심 정의:
            center_x = ox*sx + win_w/2
            center_y = oy*sy + win_h/2
        역으로:
            ox ≈ (x - win_w/2)/sx
            oy ≈ (y - win_h/2)/sy
        를 round하여 가장 가까운 셀을 선택한다.
    """
    if grid_w <= 0 or grid_h <= 0:
        return None

    ox = int(round((x - win_w / 2.0) / float(sx)))
    oy = int(round((y - win_h / 2.0) / float(sy)))

    if border == "skip":
        if ox < 0 or ox >= grid_w or oy < 0 or oy >= grid_h:
            return None
        return oy, ox

    ox = max(0, min(grid_w - 1, ox))
    oy = max(0, min(grid_h - 1, oy))
    return oy, ox


# ---------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------
def _cache_worker(args: Tuple) -> Optional[Dict[str, Any]]:
    """
    cache 서브커맨드에서 이미지(feature) 단위로 캐시를 생성/갱신하는 워커.

    Args:
        args:
            (replay_id, img_id, file_name, input_root, feature_ext, use_file_name,
             cache_root, win_w, win_h, sx, sy, thr, cache_dtype, rebuild)

    Returns:
        Optional[Dict]:
            캐시 생성 성공 시 간단한 로그 row(dict),
            feature가 없거나 실패하면 None
    """
    (
        replay_id,
        img_id,
        file_name,
        input_root,
        feature_ext,
        use_file_name,
        cache_root,
        win_w,
        win_h,
        sx,
        sy,
        thr,
        cache_dtype,
        rebuild,
    ) = args

    feat_path = build_feature_path_from_meta(
        input_root=input_root,
        replay_id=replay_id,
        file_name=file_name,
        img_id=img_id,
        use_file_name=use_file_name,
        feature_ext=feature_ext,
    )

    if not os.path.isfile(feat_path):
        return None

    cpath = ensure_kbrs_cache(
        replay_id=replay_id,
        feature_path=feat_path,
        cache_root=cache_root,
        win_w=win_w,
        win_h=win_h,
        sx=sx,
        sy=sy,
        thr=thr,
        cache_dtype=cache_dtype,
        rebuild=rebuild,
    )
    if cpath is None:
        return None

    # summary용(가볍게)
    return {
        "replay_id": replay_id,
        "image_id": int(img_id),
        "file_name": file_name,
        "cache_path": cpath,
    }


def _lookup_worker(args: Tuple) -> Optional[Dict[str, Any]]:
    """
    lookup 서브커맨드에서 이미지 단위로 (GT/pred 중심) KBRS 값을 뽑아 요약 row를 만드는 워커.

    Args:
        args:
            (replay_id, img_id, file_name, img_w, img_h,
             input_root, feature_ext, use_file_name,
             cache_root, win_w, win_h, sx, sy, thr, cache_dtype, rebuild,
             border, positions, source_tag)

    Returns:
        Optional[Dict]:
            per-image 요약 row(dict) 또는 None
    """
    (
        replay_id,
        img_id,
        file_name,
        img_w,
        img_h,
        input_root,
        feature_ext,
        use_file_name,
        cache_root,
        win_w,
        win_h,
        sx,
        sy,
        thr,
        cache_dtype,
        rebuild,
        border,
        positions,
        source_tag,
    ) = args

    feat_path = build_feature_path_from_meta(
        input_root=input_root,
        replay_id=replay_id,
        file_name=file_name,
        img_id=img_id,
        use_file_name=use_file_name,
        feature_ext=feature_ext,
    )
    if not os.path.isfile(feat_path):
        return None

    if not positions:
        return None

    # on_the_fly: 기존 방식(경계 부분 패치 포함) 재현
    if border == "on_the_fly":
        feat = np.load(feat_path)
        ds, cs, ms = [], [], []
        for cx, cy in positions:
            patch = extract_window(feat, cx, cy, win_w, win_h)
            d, c, m = kbrs_scores_from_patch(patch, threshold=thr)
            ds.append(d)
            cs.append(c)
            ms.append(m)

        if not ds:
            return None

        ds_arr = np.asarray(ds, dtype=np.float32)
        cs_arr = np.asarray(cs, dtype=np.float32)
        ms_arr = np.asarray(ms, dtype=np.float32)

        return {
            "replay_id": replay_id,
            "image_id": int(img_id),
            "file_name": file_name,
            "source": source_tag,
            "num_points": int(len(ds)),
            "mean_density": float(ds_arr.mean()),
            "max_density": float(ds_arr.max()),
            "mean_centeredness": float(cs_arr.mean()),
            "max_centeredness": float(cs_arr.max()),
            "mean_mixture": float(ms_arr.mean()),
            "max_mixture": float(ms_arr.max()),
        }

    # cache lookup
    cpath = ensure_kbrs_cache(
        replay_id=replay_id,
        feature_path=feat_path,
        cache_root=cache_root,
        win_w=win_w,
        win_h=win_h,
        sx=sx,
        sy=sy,
        thr=thr,
        cache_dtype=cache_dtype,
        rebuild=rebuild,
    )
    if cpath is None:
        return None

    density_map, centeredness_map, mixture_map, _meta = load_kbrs_cache(cpath)
    grid_h, grid_w = density_map.shape

    ds, cs, ms = [], [], []
    for cx, cy in positions:
        idx = xy_to_grid_index(cx, cy, win_w, win_h, sx, sy, grid_w, grid_h, border=border)
        if idx is None:
            continue
        oy, ox = idx
        ds.append(float(density_map[oy, ox]))
        cs.append(float(centeredness_map[oy, ox]))
        ms.append(float(mixture_map[oy, ox]))

    if not ds:
        return None

    ds_arr = np.asarray(ds, dtype=np.float32)
    cs_arr = np.asarray(cs, dtype=np.float32)
    ms_arr = np.asarray(ms, dtype=np.float32)

    return {
        "replay_id": replay_id,
        "image_id": int(img_id),
        "file_name": file_name,
        "source": source_tag,
        "num_points": int(len(ds)),
        "mean_density": float(ds_arr.mean()),
        "max_density": float(ds_arr.max()),
        "mean_centeredness": float(cs_arr.mean()),
        "max_centeredness": float(cs_arr.max()),
        "mean_mixture": float(ms_arr.mean()),
        "max_mixture": float(ms_arr.max()),
    }


# ---------------------------------------------------------------------
# CLI parsing (subcommands)
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    서브커맨드(cache/lookup)를 포함한 CLI 인자를 파싱한다.

    Returns:
        argparse.Namespace: 파싱 결과

    Subcommands:
        cache:
            replay들의 모든 image(feature)에 대해 KBRS cache(.npz)를 만든다.
        lookup:
            GT 또는 pred bbox 중심 좌표에서 KBRS 값을 cache로부터 lookup하여 CSV를 만든다.
    """
    p = argparse.ArgumentParser(description="KBRS cache/lookup CLI (subcommands)")
    sub = p.add_subparsers(dest="command", required=True)

    # 공통
    def add_common(p2: argparse.ArgumentParser) -> None:
        p2.add_argument("--replays", type=str, nargs="+", required=True, help="리플레이 ID 목록")
        p2.add_argument("--input-root", default="/workspace/data/input/dst", help="입력 feature .npy 루트")
        p2.add_argument("--label-root", default="/workspace/data/label/dst", help="GT COCO 루트")
        p2.add_argument("--label-method", default="all_correct", help="라벨링 방법(파일명)")
        p2.add_argument("--window", default="20,12", help="윈도우 (w,h)")
        p2.add_argument("--stride", default="1,1", help="stride (sx,sy)")
        p2.add_argument("--threshold", type=float, default=0.0, help="binary threshold")
        p2.add_argument("--max-frames", type=int, default=0, help="0보다 크면 images 리스트 앞에서 이 개수만 사용")
        p2.add_argument("--use-file-name", action="store_true", help="feature 파일명을 image['file_name'] 기반으로 가정")
        p2.add_argument("--feature-ext", default=".npy", help="feature 파일 확장자")
        p2.add_argument("--cache-root", default=None, help="캐시 루트(미지정 시 input-root 기반 기본값)")
        p2.add_argument("--rebuild-cache", action="store_true", help="캐시가 있어도 강제로 재생성")
        p2.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16", help="캐시 저장 dtype")
        p2.add_argument("--num-workers", type=int, default=0, help="worker 수(0이면 cpu_count())")

    # cache command
    pc = sub.add_parser("cache", help="KBRS cache 생성/갱신")
    add_common(pc)
    pc.add_argument("--csv-out", default=None, help="(선택) cache 생성 로그를 CSV로 저장(디렉터리/파일 둘 다 가능)")

    # lookup command
    pl = sub.add_parser("lookup", help="GT 또는 pred bbox 중심에서 KBRS 값을 lookup")
    add_common(pl)
    pl.add_argument("--source", choices=["gt", "pred"], required=True, help="좌표 source (gt 또는 pred)")
    pl.add_argument("--pred-root", default="/workspace/predictions", help="prediction 루트")
    pl.add_argument("--model-name", type=str, default=None, help="source=pred일 때 모델 이름")
    pl.add_argument("--epoch", type=int, default=None, help="source=pred일 때 epoch 번호")
    pl.add_argument(
        "--border",
        choices=["clamp", "skip", "on_the_fly"],
        default="clamp",
        help=(
            "경계 처리.\n"
            "  clamp: valid grid 범위로 clamp 후 cache lookup(기본)\n"
            "  skip: valid grid 밖은 버림\n"
            "  on_the_fly: 기존 방식(경계 부분 패치 포함)으로 직접 계산(느림)"
        ),
    )
    pl.add_argument("--csv-out", required=True, help="lookup 결과 CSV 출력 경로(디렉터리 또는 파일)")

    return p.parse_args()


# ---------------------------------------------------------------------
# Command runners
# ---------------------------------------------------------------------
def _iter_images(coco_gt: COCO, max_frames: int) -> List[dict]:
    """
    COCO GT에서 images 리스트를 꺼내고 max_frames를 적용한다.

    Args:
        coco_gt: GT COCO 객체
        max_frames: 0이면 전체, 0보다 크면 앞에서 해당 개수만 사용

    Returns:
        images 리스트
    """
    images = list(coco_gt.dataset.get("images", []))
    if max_frames > 0:
        images = images[:max_frames]
    return images


def run_cache(args: argparse.Namespace) -> None:
    """
    cache 서브커맨드를 실행한다.

    동작:
        - replay별로 GT COCO를 로드해서 images 목록을 얻는다(파일명/이미지 id 사용 목적).
        - 각 image에 대해 feature 경로를 만들고 KBRS cache(.npz)를 ensure(없으면 생성).
        - 옵션으로 생성 로그를 CSV로 남길 수 있다.

    Notes:
        - cache 계산은 GT/model과 무관하므로, GT COCO는 "images 메타"만 얻기 위해 사용한다.
    """
    win_w, win_h = map(int, args.window.split(","))
    sx, sy = map(int, args.stride.split(","))

    cache_root = args.cache_root or default_cache_root(args.input_root)

    all_rows: List[Dict[str, Any]] = []

    for replay_id in map(str, args.replays):
        print(f"\n[cache][Replay] {replay_id}")
        coco_gt = load_coco_gt(args.label_root, replay_id, args.label_method)
        images = _iter_images(coco_gt, args.max_frames)
        print(f"[Info] num_images={len(images)}")

        tasks: List[Tuple] = []
        for img in images:
            img_id = int(img["id"])
            file_name = img.get("file_name", "")
            tasks.append(
                (
                    replay_id,
                    img_id,
                    file_name,
                    args.input_root,
                    args.feature_ext,
                    args.use_file_name,
                    cache_root,
                    win_w,
                    win_h,
                    sx,
                    sy,
                    args.threshold,
                    args.cache_dtype,
                    args.rebuild_cache,
                )
            )

        rows: List[Dict[str, Any]] = []

        n_workers = args.num_workers or cpu_count()
        if n_workers <= 1:
            for idx, t in enumerate(tasks, 1):
                r = _cache_worker(t)
                if r is not None:
                    rows.append(r)
                if idx % 500 == 0:
                    print(f"[cache][{replay_id}] processed {idx}/{len(tasks)}")
        else:
            print(f"[Info] Using {n_workers} workers (tasks={len(tasks)})")
            with Pool(processes=n_workers) as pool:
                for idx, r in enumerate(pool.imap_unordered(_cache_worker, tasks), 1):
                    if r is not None:
                        rows.append(r)
                    if idx % 500 == 0:
                        print(f"[cache][{replay_id}] processed {idx}/{len(tasks)}")

        print(f"[cache][{replay_id}] cache created/verified for {len(rows)} images.")
        all_rows.extend(rows)

        # per-replay csv-out 디렉터리 처리
        if args.csv_out and os.path.isdir(args.csv_out):
            out_path = os.path.join(args.csv_out, f"kbrs_cache_{replay_id}.csv")
            pd.DataFrame(rows).sort_values(["image_id"]).to_csv(out_path, index=False)
            print(f"[Info] saved cache log to: {out_path}")

    # combined csv-out 파일 처리
    if args.csv_out and not os.path.isdir(args.csv_out):
        out_dir = os.path.dirname(args.csv_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(all_rows).sort_values(["replay_id", "image_id"]).to_csv(args.csv_out, index=False)
        print(f"[Info] saved combined cache log to: {args.csv_out}")


def run_lookup(args: argparse.Namespace) -> None:
    """
    lookup 서브커맨드를 실행한다.

    동작:
        - replay별로 GT COCO를 로드해서 images 메타(+GT ann) 또는 pred를 읽는다.
        - 각 image에 대해 (GT bbox 중심) 또는 (pred bbox 중심) 좌표 리스트를 만든다.
        - KBRS cache를 ensure한 뒤(border 옵션에 따라 cache lookup 또는 on-the-fly) 값을 추출한다.
        - per-image 요약 row를 CSV로 저장한다.
    """
    win_w, win_h = map(int, args.window.split(","))
    sx, sy = map(int, args.stride.split(","))

    cache_root = args.cache_root or default_cache_root(args.input_root)

    # pred 설정 검증
    preds_by_img_all: Dict[str, Dict[int, List[dict]]] = {}
    model_tag = None
    if args.source == "pred":
        if not args.model_name or args.epoch is None:
            raise ValueError("lookup --source pred 인 경우 --model-name 과 --epoch 가 필요합니다.")
        model_tag = f"{args.model_name}_e{args.epoch}"

    all_rows: List[Dict[str, Any]] = []

    for replay_id in map(str, args.replays):
        print(f"\n[lookup][Replay] {replay_id}")
        coco_gt = load_coco_gt(args.label_root, replay_id, args.label_method)
        images = _iter_images(coco_gt, args.max_frames)
        print(f"[Info] num_images={len(images)}")

        preds_by_img: Optional[Dict[int, List[dict]]] = None
        if args.source == "pred":
            preds_by_img = load_coco_preds(
                pred_root=args.pred_root,
                model_name=args.model_name,
                epoch=args.epoch,
                replay_id=replay_id,
                label_method=args.label_method,
            )

        tasks: List[Tuple] = []

        for img in images:
            img_id = int(img["id"])
            file_name = img.get("file_name", "")
            img_w = int(img.get("width", 0))
            img_h = int(img.get("height", 0))

            positions: List[Tuple[float, float]] = []

            if args.source == "gt":
                ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
                anns = coco_gt.loadAnns(ann_ids) if ann_ids else []
                if not anns:
                    continue
                positions = [centroid_from_coco_ann(ann, img_w, img_h) for ann in anns]
                if not positions:
                    continue
                source_tag = "gt"
            else:
                assert preds_by_img is not None
                dets = preds_by_img.get(img_id, [])
                if not dets:
                    continue
                positions = [
                    centroid_from_coco_ann(det, img_w, img_h)
                    for det in dets
                    if "bbox" in det and det["bbox"] is not None
                ]
                if not positions:
                    continue
                source_tag = f"pred:{model_tag}"

            tasks.append(
                (
                    replay_id,
                    img_id,
                    file_name,
                    img_w,
                    img_h,
                    args.input_root,
                    args.feature_ext,
                    args.use_file_name,
                    cache_root,
                    win_w,
                    win_h,
                    sx,
                    sy,
                    args.threshold,
                    args.cache_dtype,
                    args.rebuild_cache,
                    args.border,
                    positions,
                    source_tag,
                )
            )

        if not tasks:
            print(f"[Info] No tasks for replay={replay_id}.")
            continue

        rows: List[Dict[str, Any]] = []

        n_workers = args.num_workers or cpu_count()
        if n_workers <= 1:
            for idx, t in enumerate(tasks, 1):
                r = _lookup_worker(t)
                if r is not None:
                    rows.append(r)
                if idx % 200 == 0:
                    print(f"[lookup][{replay_id}] processed {idx}/{len(tasks)}")
        else:
            print(f"[Info] Using {n_workers} workers (tasks={len(tasks)})")
            with Pool(processes=n_workers) as pool:
                for idx, r in enumerate(pool.imap_unordered(_lookup_worker, tasks), 1):
                    if r is not None:
                        rows.append(r)
                    if idx % 200 == 0:
                        print(f"[lookup][{replay_id}] processed {idx}/{len(tasks)}")

        print(f"[lookup][{replay_id}] images processed: {len(rows)}")
        all_rows.extend(rows)

        # csv-out이 디렉터리면 replay별 저장
        if os.path.isdir(args.csv_out):
            out_path = os.path.join(args.csv_out, f"kbrs_lookup_{args.source}_{replay_id}.csv")
            pd.DataFrame(rows).sort_values(["image_id"]).to_csv(out_path, index=False)
            print(f"[Info] saved per-replay lookup csv to: {out_path}")

    # csv-out이 파일이면 전체 합쳐 저장
    if not os.path.isdir(args.csv_out):
        out_dir = os.path.dirname(args.csv_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(all_rows).sort_values(["replay_id", "image_id"]).to_csv(args.csv_out, index=False)
        print(f"[Info] saved combined lookup csv to: {args.csv_out}")


def main() -> None:
    """
    엔트리 포인트.

    흐름:
        - parse_args()로 command를 읽고
        - cache면 run_cache(), lookup이면 run_lookup()를 호출한다.
    """
    args = parse_args()
    if args.command == "cache":
        run_cache(args)
    elif args.command == "lookup":
        run_lookup(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
