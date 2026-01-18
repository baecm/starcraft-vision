# cache.py
from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO
from multiprocessing import Pool, cpu_count

from utils.logger import Logger

# ============================================================
# 채널/컴포넌트 정의 (config.py 기반)
# ============================================================
CHANNEL_NAMES_11 = [
    "Player_1_Worker",    # 0
    "Player_1_Ground",    # 1
    "Player_1_Air",       # 2
    "Player_1_Building",  # 3
    "Player_2_Worker",    # 4
    "Player_2_Ground",    # 5
    "Player_2_Air",       # 6
    "Player_2_Building",  # 7
    "Resource",           # 8
    "Vision",             # 9
    "Terrain",            # 10
]

COMPONENT_CHANNEL_MAP_11: Dict[str, List[int]] = {
    "worker":   [0, 4],
    "ground":   [1, 5],
    "air":      [2, 6],
    "building": [3, 7],
    "resource": [8],
    "vision":   [9],
    "terrain":  [10],
}


# ============================================================
# KBRSConvScorer import (프로젝트 코드 재사용)
# ============================================================
def _import_kbrs_scorer():
    """
    프로젝트 내 `KBRSConvScorer`를 import한다.

    - 목적: 캐시 계산을 학습/추론에서 쓰는 scorer 정의와 1:1로 맞추기.
    - 실패 시: ImportError를 발생시켜서 사용자가 PYTHONPATH/경로를 맞추도록 유도.
    """
    from model.kbrs import KBRSConvScorer  # 프로젝트 경로에 맞게 조정 가능
    return KBRSConvScorer


# ============================================================
# 경로/COCO 로딩 유틸
# ============================================================
def load_coco_gt(label_root: str, replay_id: str, label_method: str) -> COCO:
    """
    GT COCO 파일을 로드한다.

    경로 규칙:
      {label_root}/{replay_id}.rep/{label_method}.json
    """
    gt_dir = os.path.join(label_root, f"{replay_id}.rep")
    gt_path = os.path.join(gt_dir, f"{label_method}.json")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Ground truth (COCO) file not found: {gt_path}")
    return COCO(gt_path)


def load_coco_preds(
    pred_root: str,
    model_name: str,
    epoch: int,
    replay_id: str,
    label_method: str,
) -> Dict[int, List[dict]]:
    """
    모델 prediction COCO dets(JSON)를 로드해서 image_id별로 묶어 반환한다.

    경로 규칙:
      {pred_root}/{model_name}/model_{epoch}/{replay_id}.rep/{label_method}.json

    반환:
      preds_by_img[image_id] = [det, det, ...]
    """
    pred_dir = os.path.join(
        pred_root, model_name, f"model_{epoch}", f"{replay_id}.rep"
    )
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
    feature 경로를 구성한다.

    규칙:
      {input_root}/{replay_id}.rep/{file_name or id}{feature_ext}
    """
    rep_dir = os.path.join(input_root, f"{replay_id}.rep")
    if use_file_name and file_name:
        base, _ext = os.path.splitext(file_name)
        fname = base + feature_ext
    else:
        fname = f"{img_id}{feature_ext}"
    return os.path.join(rep_dir, fname)


def build_cache_path(
    cache_root: str,
    replay_id: str,
    file_name: str,
    img_id: int,
    use_file_name: bool,
    cache_ext: str = ".npz",
) -> str:
    """
    캐시 파일 경로를 구성한다.

    기본 규칙:
      {cache_root}/{replay_id}.rep/{file_name or id}{cache_ext}
    """
    rep_dir = os.path.join(cache_root, f"{replay_id}.rep")
    if use_file_name and file_name:
        base, _ext = os.path.splitext(file_name)
        fname = base + cache_ext
    else:
        fname = f"{img_id}{cache_ext}"
    return os.path.join(rep_dir, fname)


def centroid_from_coco_ann(ann: dict, img_w: int, img_h: int) -> Tuple[float, float]:
    """
    COCO annotation(예측/GT)에 대해 bbox 중심을 픽셀 좌표로 반환한다.

    - bbox가 없으면 이미지 중앙으로 fallback한다.
    """
    if "bbox" in ann and ann["bbox"] is not None:
        x, y, w, h = ann["bbox"]
        return float(x + w / 2.0), float(y + h / 2.0)
    return float(img_w) / 2.0, float(img_h) / 2.0


# ============================================================
# 채널 선택/재매핑
# ============================================================
def build_selected_channels_11(include_components: List[str]) -> List[int]:
    """
    include_components 목록을 기반으로, 원본 11채널에서 사용할 채널 인덱스 리스트를 만든다.

    예:
      include_components = [worker, ground, air, building, vision]
      -> [0,1,2,3,4,5,6,7,9]
    """
    chs: List[int] = []
    for comp in include_components:
        if comp not in COMPONENT_CHANNEL_MAP_11:
            raise ValueError(f"Unknown component: {comp}")
        chs.extend(COMPONENT_CHANNEL_MAP_11[comp])
    # 중복 제거 + 순서 유지
    chs = list(dict.fromkeys(chs))
    return chs


def remap_indices(old_to_new: Dict[int, int], idxs: List[int]) -> List[int]:
    """
    원본 인덱스 리스트(idxs)를 old_to_new 매핑으로 재인덱싱한다.

    - old_to_new[old_idx] = new_idx
    """
    out: List[int] = []
    for i in idxs:
        if i not in old_to_new:
            raise ValueError(f"Index {i} not in old_to_new remap.")
        out.append(old_to_new[i])
    return out


# ============================================================
# cache 계산 (워커)
# ============================================================
@dataclass(frozen=True)
class CacheTask:
    """
    캐시 생성 작업 단위(프레임 단위).

    - mode/cache 공통 메타를 포함하되, worker는 frame별로 독립 실행 가능하도록 구성한다.
    """
    replay_id: str
    image_id: int
    file_name: str
    input_root: str
    label_root: str
    label_method: str
    feature_ext: str
    use_file_name: bool

    cache_root: str
    cache_dtype: str
    compress: bool
    rebuild_cache: bool

    include_components: Tuple[str, ...]
    region_size: Tuple[int, int]      # (kH, kW)  ※ conv2d 기준
    score_stride: int
    downsample_before: Optional[Dict[str, Any]]

    score_weights: Dict[str, float]
    projections_9: Dict[str, List[int]]  # 9ch 기준 인덱스
    mixture_mode: str
    mixture_power: float

    gate_channels_9: List[int]
    gate_reduce: str
    gate_gain: float


def _cache_worker(t: CacheTask) -> Dict[str, Any]:
    """
    단일 프레임에 대해 KBRS cache(npz)를 생성한다.

    반환 dict는 tqdm/postfix 및 요약 집계에 사용한다:
      - status: "hit" | "build" | "skip_missing_feature" | "error"
      - path: cache_path
      - mean/max: (optional) 간단한 통계
    """
    try:
        Logger.info(f"[cache_worker] start replay={t.replay_id} image_id={t.image_id} file_name='{t.file_name}'")

        feat_path = build_feature_path_from_meta(
            input_root=t.input_root,
            replay_id=t.replay_id,
            file_name=t.file_name,
            img_id=t.image_id,
            use_file_name=t.use_file_name,
            feature_ext=t.feature_ext,
        )
        Logger.info(f"[cache_worker] feat_path={feat_path}")

        if not os.path.isfile(feat_path):
            Logger.warn(f"[cache_worker] missing feature -> skip: {feat_path}")
            return {"status": "skip_missing_feature", "path": feat_path}

        cache_path = build_cache_path(
            cache_root=t.cache_root,
            replay_id=t.replay_id,
            file_name=t.file_name,
            img_id=t.image_id,
            use_file_name=t.use_file_name,
            cache_ext=".npz",
        )
        Logger.info(f"[cache_worker] cache_path={cache_path}")

        if (not t.rebuild_cache) and os.path.isfile(cache_path):
            Logger.info("[cache_worker] cache hit")
            return {"status": "hit", "path": cache_path}

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        Logger.info("[cache_worker] np.load feature")
        feat = np.load(feat_path, allow_pickle=True)
        Logger.info(f"[cache_worker] feat.shape={getattr(feat, 'shape', None)} feat.dtype={getattr(feat, 'dtype', None)} type={type(feat)}")

        if not hasattr(feat, "ndim") or feat.ndim != 3:
            Logger.error(f"[cache_worker] bad feat ndim/shape: {feat_path} shape={getattr(feat, 'shape', None)}")
            return {"status": "error", "path": feat_path, "error": f"bad feat shape: {getattr(feat, 'shape', None)}"}

        # HWC(128,128,11) 저장 케이스를 자동 보정
        if feat.shape[0] != 11 and feat.shape[-1] == 11:
            Logger.warn(f"[cache_worker] detected HWC, transpose -> CHW: {feat.shape}")
            feat = np.transpose(feat, (2, 0, 1))
            Logger.info(f"[cache_worker] after transpose feat.shape={feat.shape}")

        # 이미 9채널로 저장된 케이스도 허용(개발단계 편의)
        if feat.shape[0] not in (11, 9):
            Logger.error(f"[cache_worker] unexpected channel count: C={feat.shape[0]} path={feat_path}")
            return {"status": "error", "path": feat_path, "error": f"unexpected C={feat.shape[0]}"}

        include_components = list(t.include_components)

        if feat.shape[0] == 11:
            sel11 = build_selected_channels_11(include_components)
            Logger.info(f"[cache_worker] selected sel11={sel11}")
            feat9 = feat[sel11, :, :]
            Logger.info(f"[cache_worker] feat9.shape={feat9.shape}")
        else:
            # feat가 이미 9ch이면 그대로 사용(assume 학습 입력 채널 순서)
            sel11 = None
            feat9 = feat
            Logger.warn("[cache_worker] feature already has 9 channels; skipping 11->9 selection")

        Logger.info("[cache_worker] import KBRSConvScorer")
        KBRSConvScorer = _import_kbrs_scorer()

        Logger.info("[cache_worker] instantiate scorer")
        scorer = KBRSConvScorer(
            region_size=t.region_size,
            weights=t.score_weights,
            projections=t.projections_9,
            mixture_mode=t.mixture_mode,
            mixture_power=t.mixture_power,
            score_stride=t.score_stride,
            downsample_before=t.downsample_before,
        )
        scorer.eval()

        import torch

        x = torch.from_numpy(feat9).unsqueeze(0).float()  # (1,9,H,W)
        Logger.info(f"[cache_worker] torch input shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")

        with torch.no_grad():
            score_map, comp_maps = scorer(x)

        Logger.info(f"[cache_worker] score_map.shape={tuple(score_map.shape)} comp_keys={list(comp_maps.keys())}")

        def _to_np(a: torch.Tensor) -> np.ndarray:
            arr = a.detach().cpu().numpy()
            if t.cache_dtype == "float16":
                return arr.astype(np.float16)
            if t.cache_dtype == "float32":
                return arr.astype(np.float32)
            return arr

        score_np = _to_np(score_map[0])
        den_np = _to_np(comp_maps.get("density", torch.zeros_like(score_map))[0])
        cen_np = _to_np(comp_maps.get("centeredness", torch.zeros_like(score_map))[0])
        mix_np = _to_np(comp_maps.get("mixture", torch.zeros_like(score_map))[0])

        meta = {
            "replay_id": t.replay_id,
            "image_id": int(t.image_id),
            "file_name": t.file_name,
            "feat_path": feat_path,
            "feat_shape": list(getattr(feat, "shape", [])),
            "selected_channels_11": sel11,
            "selected_channel_names_11": [CHANNEL_NAMES_11[i] for i in sel11] if sel11 is not None else None,
            "region_size_kh_kw": list(t.region_size),
            "score_stride": int(t.score_stride),
            "downsample_before": t.downsample_before,
            "score_weights": t.score_weights,
            "projections_9": t.projections_9,
            "mixture_mode": t.mixture_mode,
            "mixture_power": float(t.mixture_power),
            "gate_channels_9": t.gate_channels_9,
            "gate_reduce": t.gate_reduce,
            "gate_gain": float(t.gate_gain),
            "note": "score_map is weighted sum of density/centeredness/mixture (gate not applied here).",
        }
        meta_json = json.dumps(meta, ensure_ascii=False)

        Logger.info(f"[cache_worker] saving npz -> {cache_path} compress={t.compress} dtype={t.cache_dtype}")
        if t.compress:
            np.savez_compressed(
                cache_path,
                score=score_np,
                density=den_np,
                centeredness=cen_np,
                mixture=mix_np,
                meta_json=np.array([meta_json], dtype=object),
            )
        else:
            np.savez(
                cache_path,
                score=score_np,
                density=den_np,
                centeredness=cen_np,
                mixture=mix_np,
                meta_json=np.array([meta_json], dtype=object),
            )

        Logger.info("[cache_worker] build done")
        return {
            "status": "build",
            "path": cache_path,
            "mean_score": float(np.mean(score_np)),
            "max_score": float(np.max(score_np)),
        }

    except Exception:
        import traceback
        tb = traceback.format_exc()
        Logger.error("[cache_worker] exception\n", tb)
        return {"status": "error", "path": "", "error": tb}


# ============================================================
# lookup 계산 (워커)
# ============================================================
@dataclass(frozen=True)
class LookupTask:
    """
    lookup 작업 단위(프레임 단위).

    - cache(npz)에서 score/comp 맵을 읽고,
      GT 또는 pred의 bbox 중심 좌표들을 맵으로 매핑해 값을 집계한다.
    """
    replay_id: str
    image_id: int
    file_name: str
    img_w: int
    img_h: int

    cache_root: str
    use_file_name: bool

    region_size: Tuple[int, int]  # (kH,kW)
    score_stride: int
    border_mode: str              # clamp|skip

    source: str                   # "gt" | "pred"
    positions: List[Tuple[float, float]]  # (x,y) in pixel coords
    label_method: str             # for row metadata


def xy_to_grid_index(
    x: float,
    y: float,
    H: int,
    W: int,
    kH: int,
    kW: int,
    stride: int,
    border_mode: str,
) -> Optional[Tuple[int, int]]:
    """
    픽셀 좌표(x,y)를 KBRS score_map의 grid index(oy,ox)로 매핑한다.

    conv2d(valid) 기준:
      out_h = (H - kH)//stride + 1
      out_w = (W - kW)//stride + 1

    윈도우의 "중심" 기준으로 가장 가까운 셀을 찾는다:
      center_x = ox*stride + kW/2
      center_y = oy*stride + kH/2

    border_mode:
      - "clamp": 범위 밖이면 가장 가까운 유효 인덱스로 clamp
      - "skip" : 범위 밖이면 None
    """
    out_h = (H - kH) // stride + 1
    out_w = (W - kW) // stride + 1
    if out_h <= 0 or out_w <= 0:
        return None

    # 중심 기준 역변환
    ox = int(round((x - (kW / 2.0)) / float(stride)))
    oy = int(round((y - (kH / 2.0)) / float(stride)))

    if border_mode == "skip":
        if ox < 0 or oy < 0 or ox >= out_w or oy >= out_h:
            return None
        return oy, ox

    # clamp (default)
    ox = max(0, min(out_w - 1, ox))
    oy = max(0, min(out_h - 1, oy))
    return oy, ox


def _load_cache_npz(cache_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    cache npz에서 score/density/centeredness/mixture 및 meta(dict)를 읽는다.

    meta_json은 object 배열로 저장되어 있으므로 allow_pickle=True가 필요하다.
    """
    with np.load(cache_path, allow_pickle=True) as z:
        score = z["score"]
        den = z["density"]
        cen = z["centeredness"]
        mix = z["mixture"]
        meta_json = str(z["meta_json"][0])
    meta = json.loads(meta_json)
    return score, den, cen, mix, meta


def _lookup_worker(t: LookupTask) -> Optional[Dict[str, Any]]:
    """
    단일 프레임에 대해 lookup을 수행하고, per-image row를 반환한다.

    반환 row는 CSV로 저장될 수 있는 dict 형태이다.
    """
    cache_path = build_cache_path(
        cache_root=t.cache_root,
        replay_id=t.replay_id,
        file_name=t.file_name,
        img_id=t.image_id,
        use_file_name=t.use_file_name,
        cache_ext=".npz",
    )
    if not os.path.isfile(cache_path):
        return None

    score, den, cen, mix, meta = _load_cache_npz(cache_path)

    # feature 원본 크기(128x128)를 기준으로 grid 매핑한다.
    # 여기서는 meta에 원본 H,W를 저장하지 않았으므로, label json에서 받은 img_w/img_h를 사용한다.
    # (학습에서 항상 128x128이면 문제 없음. 다를 경우 meta에 ORIGIN_SHAPE 저장을 권장.)
    H = int(t.img_h)
    W = int(t.img_w)
    kH, kW = t.region_size
    stride = int(t.score_stride)

    vals_score = []
    vals_den = []
    vals_cen = []
    vals_mix = []

    for (x, y) in t.positions:
        idx = xy_to_grid_index(
            x=x, y=y,
            H=H, W=W,
            kH=kH, kW=kW,
            stride=stride,
            border_mode=t.border_mode,
        )
        if idx is None:
            continue
        oy, ox = idx
        vals_score.append(float(score[oy, ox]))
        vals_den.append(float(den[oy, ox]))
        vals_cen.append(float(cen[oy, ox]))
        vals_mix.append(float(mix[oy, ox]))

    if not vals_score:
        return None

    row = {
        "replay_id": t.replay_id,
        "image_id": int(t.image_id),
        "file_name": t.file_name,
        "source": t.source,
        "label_method": t.label_method,
        "num_points": int(len(vals_score)),
        "mean_score": float(np.mean(vals_score)),
        "max_score": float(np.max(vals_score)),
        "mean_density": float(np.mean(vals_den)),
        "max_density": float(np.max(vals_den)),
        "mean_centeredness": float(np.mean(vals_cen)),
        "max_centeredness": float(np.max(vals_cen)),
        "mean_mixture": float(np.mean(vals_mix)),
        "max_mixture": float(np.max(vals_mix)),
    }
    return row


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    """
    커맨드라인 인자를 파싱한다.

    서브커맨드:
      - cache  : 프레임별 KBRS 맵(npz) 생성
      - lookup : GT/pred 좌표를 기반으로 캐시에서 값 추출하여 CSV 저장
    """
    p = argparse.ArgumentParser(description="KBRS cache/lookup tool (conv-based, 11->9 channel selection).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # -------- cache --------
    pc = sub.add_parser("cache", help="Build per-frame KBRS cache (.npz).")
    pc.add_argument("--replays", type=str, nargs="+", required=True)

    pc.add_argument("--data-root", default="/workspace/data")
    pc.add_argument("--input-root", default="/workspace/data/input/dst")
    pc.add_argument("--label-root", default="/workspace/data/label/dst")
    pc.add_argument("--label-method", default="all_correct")

    pc.add_argument("--feature-ext", default=".npy")
    pc.add_argument("--use-file-name", action="store_true")

    pc.add_argument("--cache-root", default=None)
    pc.add_argument("--cache-dtype", default="float16", choices=["float16", "float32"])
    pc.add_argument("--compress", action="store_true")
    pc.add_argument("--rebuild-cache", action="store_true")

    pc.add_argument(
        "--include-components",
        nargs="+",
        default=["worker", "ground", "air", "building", "vision"],
        help="11채널에서 사용할 컴포넌트 목록 (예: worker ground air building vision).",
    )

    # KBRS params (config.py KBRS_PARAMS 기반)
    pc.add_argument("--region-size", default="20,12", help="(kW,kH). 예: 20,12 (w,h)")
    pc.add_argument("--score-stride", type=int, default=1)
    pc.add_argument("--downsample-before", default=None, help='예: {"type":"avg","stride":2} (JSON 문자열)')

    pc.add_argument("--w-density", type=float, default=0.3)
    pc.add_argument("--w-mixture", type=float, default=3.0)
    pc.add_argument("--w-centeredness", type=float, default=0.3)

    pc.add_argument("--proj-A", default="0,1,2,3", help="9ch 기준 A projection 채널 인덱스")
    pc.add_argument("--proj-B", default="4,5,6,7", help="9ch 기준 B projection 채널 인덱스")

    pc.add_argument("--mixture-mode", default="confusion", choices=["confusion", "entropy"])
    pc.add_argument("--mixture-power", type=float, default=2.0)

    pc.add_argument("--gate-channels", default="8", help="9ch 기준 gate 채널(보통 vision=8)")
    pc.add_argument("--gate-reduce", default="mean", choices=["mean"])
    pc.add_argument("--gate-gain", type=float, default=0.8)

    pc.add_argument("--max-frames", type=int, default=0)
    pc.add_argument("--num-workers", type=int, default=0)
    pc.add_argument("--chunksize", type=int, default=1)
    pc.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="0~1 사이. replay 내 frames 중 앞에서 이 비율만 처리(개발용). 예: 0.05",
    )

    pc.add_argument(
    "--log-level",
    default="log",
    choices=["none", "log", "debug"],
    help="Logger 출력 레벨. debug면 워커 단계 로그까지 볼 수 있음.",
    )
    pc.add_argument(
        "--debug-samples",
        type=int,
        default=0,
        help="0이면 워커 단계 로그 없음. >0이면 replay별로 처음 N개 task만 상세 로그 출력.",
    )
    pc.add_argument(
        "--error-samples",
        type=int,
        default=5,
        help="replay별로 에러 메시지 샘플을 최대 N개까지 모아서 출력.",
    )


    # -------- lookup --------
    pl = sub.add_parser("lookup", help="Lookup values at GT/pred positions from cache and write CSV.")
    pl.add_argument("--replays", type=str, nargs="+", required=True)

    pl.add_argument("--data-root", default="/workspace/data")
    pl.add_argument("--label-root", default=None)
    pl.add_argument("--label-method", default="all_correct")
    pl.add_argument("--use-file-name", action="store_true")

    pl.add_argument("--cache-root", default=None)

    pl.add_argument("--source", default="gt", choices=["gt", "pred"])
    pl.add_argument("--pred-root", default="/workspace/predictions")
    pl.add_argument("--model-name", default=None)
    pl.add_argument("--epoch", type=int, default=None)

    pl.add_argument("--border-mode", default="clamp", choices=["clamp", "skip"])
    pl.add_argument("--csv-out", required=True)

    pl.add_argument("--region-size", default="20,12", help="(kH,kW) conv 기준. 예: 20,12")
    pl.add_argument("--score-stride", type=int, default=1)
    pl.add_argument("--max-frames", type=int, default=0)
    pl.add_argument("--num-workers", type=int, default=0)
    pl.add_argument("--chunksize", type=int, default=1)
    pl.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="0~1 사이. replay 내 frames 중 앞에서 이 비율만 처리(개발용). 예: 0.05",
    )

    return p.parse_args()


def _parse_int_list(s: str) -> List[int]:
    """
    '0,1,2' 형태 문자열을 int 리스트로 파싱한다.
    """
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def run_cache(args: argparse.Namespace) -> None:
    """
    cache 서브커맨드를 실행한다.

    - COCO GT의 images 리스트를 기준으로 프레임을 순회하며,
      feature(.npy)를 읽어 11->9 채널 선택 후 KBRSConvScorer로 맵을 계산하고 npz로 저장한다.
    - 진행도(tqdm)와 hit/build/skip/error 카운트를 표시한다.
    """    
    input_root = args.input_root or os.path.join(args.data_root, "input/dst")
    label_root = args.label_root or os.path.join(args.data_root, "label/dst")
    cache_root = args.cache_root or os.path.join(input_root, "__kbrs_cache__")

    kW, kH = map(int, args.region_size.split(","))
    score_weights = {"density": args.w_density, "mixture": args.w_mixture, "centeredness": args.w_centeredness}

    downsample_before = None
    if args.downsample_before:
        downsample_before = json.loads(args.downsample_before)

    projections_9 = {"A": _parse_int_list(args.proj_A), "B": _parse_int_list(args.proj_B)}
    gate_channels_9 = _parse_int_list(args.gate_channels)

    Logger.set_level(args.log_level)
    Logger.info(f"[cache] input_root={input_root}")
    Logger.info(f"[cache] label_root={label_root}")
    Logger.info(f"[cache] cache_root={cache_root}")
    Logger.info(f"[cache] include_components={args.include_components}")
    Logger.info(f"[cache] region_size(kW,kH)=({kW},{kH}) -> internal(kH,kW)=({kH},{kW}) score_stride={args.score_stride}")
    Logger.info(f"[cache] projections_9={projections_9} mixture_mode={args.mixture_mode} mixture_power={args.mixture_power}")

    for replay_id in args.replays:
        replay_id = str(replay_id)
        coco = load_coco_gt(label_root=label_root, replay_id=replay_id, label_method=args.label_method)
        images = list(coco.dataset.get("images", []))
        if args.max_frames > 0:
            images = images[: args.max_frames]
        # 개발용: replay 내 일부만 처리
        if args.sample_ratio < 1.0:
            if not (0.0 < args.sample_ratio <= 1.0):
                raise ValueError("--sample-ratio must be in (0, 1].")
            n = max(1, int(len(images) * args.sample_ratio))
            images = images[:n]

        tasks: List[CacheTask] = []
        for i, img in enumerate(images):
            img_id = int(img["id"])
            file_name = img.get("file_name", "")
            tasks.append(
                CacheTask(
                    replay_id=replay_id,
                    image_id=img_id,
                    file_name=file_name,
                    input_root=input_root,
                    label_root=label_root,
                    label_method=args.label_method,
                    feature_ext=args.feature_ext,
                    use_file_name=args.use_file_name,
                    cache_root=cache_root,
                    cache_dtype=args.cache_dtype,
                    compress=bool(args.compress),
                    rebuild_cache=bool(args.rebuild_cache),
                    include_components=tuple(args.include_components),
                    region_size=(kH, kW),
                    score_stride=int(args.score_stride),
                    downsample_before=downsample_before,
                    score_weights=score_weights,
                    projections_9=projections_9,
                    mixture_mode=str(args.mixture_mode),
                    mixture_power=float(args.mixture_power),
                    gate_channels_9=gate_channels_9,
                    gate_reduce=str(args.gate_reduce),
                    gate_gain=float(args.gate_gain),
                )
            )

        if not tasks:
            print(f"[cache] replay={replay_id}: no tasks")
            continue

        n_workers = args.num_workers or cpu_count()
        chunksize = max(1, int(args.chunksize))

        counters = {"hit": 0, "build": 0, "skip_missing_feature": 0, "error": 0}
        error_samples: List[str] = []

        desc = f"[cache] {replay_id}"
        if n_workers == 1:
            for t in tqdm(tasks, desc=desc, total=len(tasks)):
                r = _cache_worker(t)
                st = r.get("status", "error")
                counters[st] = counters.get(st, 0) + 1
                
        else:
            with Pool(processes=n_workers) as pool:
                it = pool.imap_unordered(_cache_worker, tasks, chunksize=chunksize)
                for r in tqdm(it, desc=desc, total=len(tasks)):
                    st = r.get("status", "error")
                    counters[st] = counters.get(st, 0) + 1

                    if st == "error":
                        msg = str(r.get("error", ""))
                        if msg and len(error_samples) < int(args.error_samples):
                            error_samples.append(msg)

        Logger.info(f"[cache] replay={replay_id} done: {counters}  cache_root={cache_root}")
        if error_samples:
            Logger.error(f"[cache] replay={replay_id} error samples (up to {args.error_samples}):")
            for e in error_samples:
                Logger.error("  -", e)

def run_lookup(args: argparse.Namespace) -> None:
    """
    lookup 서브커맨드를 실행한다.

    - GT 또는 pred bbox 중심 좌표를 구하고,
      cache(npz)의 score/comp 맵에서 해당 위치 값을 샘플링해 per-image 요약을 만들고 CSV로 저장한다.
    """
    label_root = args.label_root or os.path.join(args.data_root, "label/dst")
    cache_root = args.cache_root or os.path.join(os.path.join(args.data_root, "input/dst"), "__kbrs_cache__")

    kH, kW = map(int, args.region_size.split(","))
    if args.source == "pred":
        if not args.model_name or args.epoch is None:
            raise ValueError("source=pred 인 경우 --model-name 과 --epoch 를 지정해야 합니다.")

    all_rows: List[Dict[str, Any]] = []

    for replay_id in args.replays:
        replay_id = str(replay_id)
        coco = load_coco_gt(label_root=label_root, replay_id=replay_id, label_method=args.label_method)
        images = list(coco.dataset.get("images", []))
        if args.max_frames > 0:
            images = images[: args.max_frames]

        preds_by_img: Optional[Dict[int, List[dict]]] = None
        if args.source == "pred":
            preds_by_img = load_coco_preds(
                pred_root=args.pred_root,
                model_name=args.model_name,
                epoch=args.epoch,
                replay_id=replay_id,
                label_method=args.label_method,
            )

        tasks: List[LookupTask] = []

        for img in images:
            img_id = int(img["id"])
            file_name = img.get("file_name", "")
            img_w = int(img.get("width", 128))
            img_h = int(img.get("height", 128))

            positions: List[Tuple[float, float]] = []

            if args.source == "gt":
                ann_ids = coco.getAnnIds(imgIds=[img_id])
                anns = coco.loadAnns(ann_ids) if ann_ids else []
                if not anns:
                    continue
                positions = [centroid_from_coco_ann(a, img_w, img_h) for a in anns]

            else:
                assert preds_by_img is not None
                dets = preds_by_img.get(img_id, [])
                if not dets:
                    continue
                positions = [
                    centroid_from_coco_ann(d, img_w, img_h)
                    for d in dets
                    if "bbox" in d and d["bbox"] is not None
                ]

            if not positions:
                continue

            tasks.append(
                LookupTask(
                    replay_id=replay_id,
                    image_id=img_id,
                    file_name=file_name,
                    img_w=img_w,
                    img_h=img_h,
                    cache_root=cache_root,
                    use_file_name=args.use_file_name,
                    region_size=(kH, kW),
                    score_stride=int(args.score_stride),
                    border_mode=str(args.border_mode),
                    source=args.source,
                    positions=positions,
                    label_method=args.label_method,
                )
            )

        if not tasks:
            print(f"[lookup] replay={replay_id}: no tasks")
            continue

        n_workers = args.num_workers or cpu_count()
        chunksize = max(1, int(args.chunksize))
        desc = f"[lookup:{args.source}] {replay_id}"

        rows: List[Dict[str, Any]] = []
        if n_workers == 1:
            for t in tqdm(tasks, desc=desc, total=len(tasks)):
                r = _lookup_worker(t)
                if r is not None:
                    rows.append(r)
        else:
            with Pool(processes=n_workers) as pool:
                it = pool.imap_unordered(_lookup_worker, tasks, chunksize=chunksize)
                for r in tqdm(it, desc=desc, total=len(tasks)):
                    if r is not None:
                        rows.append(r)

        if rows:
            all_rows.extend(rows)
            print(f"[lookup] replay={replay_id}: rows={len(rows)}")

    if not all_rows:
        print("[lookup] no rows to write")
        return

    out_dir = os.path.dirname(args.csv_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(all_rows).sort_values(["replay_id", "image_id"])
    df.to_csv(args.csv_out, index=False)
    print(f"[lookup] saved: {args.csv_out}  rows={len(df)}")


def main() -> None:
    """
    엔트리포인트.

    - parse_args()로 커맨드/옵션을 읽고,
      cache 또는 lookup 실행 함수를 호출한다.
    """
    args = parse_args()
    if args.cmd == "cache":
        run_cache(args)
    elif args.cmd == "lookup":
        run_lookup(args)
    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
