# src/estimate.py
from __future__ import annotations

import os
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from multiprocessing import Pool, cpu_count

# IC metric(기존 evaluator) 및 Multi-Region Evaluator 재사용
from evaluate import eval_kernel_from_coco
from metrics.evaluator import MultiRegionEvaluator


def _debug_listdir(path: str, label: str = "") -> None:
    """
    디렉토리 존재 여부 + 내부 파일/디렉토리 목록을 출력하는 디버그용 헬퍼.
    """
    if label:
        prefix = f"[DEBUG][{label}]"
    else:
        prefix = "[DEBUG]"

    if os.path.isdir(path):
        try:
            entries = os.listdir(path)
        except Exception as e:
            print(f"{prefix} failed to listdir: {path} ({e})")
            return

        print(f"{prefix} dir exists: {path}")
        if not entries:
            print(f"{prefix}   (empty)")
        else:
            print(f"{prefix}   contains {len(entries)} entries:")
            for name in entries:
                print(f"{prefix}     - {name}")
    else:
        print(f"{prefix} dir NOT found: {path}")


# =====================================================================
# 1. KBRS metric score 계산 관련 코드 (density / centeredness / mixture)
# =====================================================================

def make_gaussian_kernel(h: int, w: int, sigma: Optional[float] = None) -> np.ndarray:
    if sigma is None:
        sigma = h / 4.0
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    dist2 = (ys - cy) ** 2 + (xs - cx) ** 2
    g = np.exp(-dist2 / (2.0 * sigma * sigma))
    return g.astype(np.float32)


def kbrs_scores_from_patch(
    patch: np.ndarray,
    threshold: float = 0.0,
) -> Tuple[float, float, float]:
    """
    KBRS patch score:
      - density: binary map mean
      - centeredness: binary map * gaussian kernel 의 weighted mean
      - mixture: density * centeredness
    """
    if patch.ndim == 3:
        patch2d = patch.max(axis=0)  # 채널 max projection
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


def extract_window(
    feat: np.ndarray,
    center_x: float,
    center_y: float,
    win_w: int,
    win_h: int,
) -> np.ndarray:
    """
    feature 맵(feat)에서 (center_x, center_y)를 중심으로 win_w x win_h 패치 잘라오기.
    feat: (C,H,W) 또는 (H,W)
    """
    if feat.ndim == 3:
        C, H, W = feat.shape
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
        else:
            return feat[0:0, 0:0]

    if feat.ndim == 3:
        return feat[:, y0:y1, x0:x1]
    else:
        return feat[y0:y1, x0:x1]


def compute_kbrs_grid(
    feat: np.ndarray,
    win_w: int,
    win_h: int,
    stride_x: int = 1,
    stride_y: int = 1,
    threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    전체 feature 맵 위를 슬라이딩 윈도우로 훑으면서
    KBRS map 3개(density/centered/mixture)를 만드는 함수.
    (mode=feature 에서 사용)
    """
    if feat.ndim == 3:
        C, H, W = feat.shape
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


# ========= KBRS용 worker / replay-level 집계 =========

def centroid_from_coco_ann(ann: dict, img_w: int, img_h: int) -> Tuple[float, float]:
    if "bbox" in ann and ann["bbox"] is not None:
        x, y, w, h = ann["bbox"]
        return float(x + w / 2.0), float(y + h / 2.0)
    return float(img_w) / 2.0, float(img_h) / 2.0


def build_feature_path_from_meta(
    input_root: str,
    replay_id: str,
    file_name: str,
    img_id: int,
    use_file_name: bool,
    feature_ext: str,
) -> str:
    rep_dir = os.path.join(input_root, f"{replay_id}.rep")
    if use_file_name and file_name:
        base, _ext = os.path.splitext(file_name)
        fname = base + feature_ext
    else:
        fname = f"{img_id}{feature_ext}"
    return os.path.join(rep_dir, fname)


def _process_image_worker(args: Tuple) -> Optional[Dict]:
    (
        mode,
        replay_id,
        img_id,
        file_name,
        img_w,
        img_h,
        input_root,
        feature_ext,
        use_file_name,
        win_w,
        win_h,
        stride_x,
        stride_y,
        threshold,
        positions,  # None (feature) or List[(cx,cy)] for gt/model
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
        print(
            f"[Warn][replay={replay_id}] feature not found for "
            f"image_id={img_id}, file_name='{file_name}': {feat_path}"
        )
        return None

    try:
        feat = np.load(feat_path)
    except Exception as e:
        print(
            f"[Error][replay={replay_id}] failed to load feature "
            f"for image_id={img_id}, path={feat_path}: {e}"
        )
        return None

    # mode=feature → 전체 grid
    if mode == "feature":
        density_map, centeredness_map, mixture_map = compute_kbrs_grid(
            feat,
            win_w=win_w,
            win_h=win_h,
            stride_x=stride_x,
            stride_y=stride_y,
            threshold=threshold,
        )
        return {
            "replay_id": replay_id,
            "image_id": img_id,
            "file_name": file_name,
            "source": source_tag,
            "num_points": 0,
            "mean_density": float(density_map.mean()),
            "max_density": float(density_map.max()),
            "mean_centeredness": float(centeredness_map.mean()),
            "max_centeredness": float(centeredness_map.max()),
            "mean_mixture": float(mixture_map.mean()),
            "max_mixture": float(mixture_map.max()),
        }

    # mode=gt / mode=model → 특정 position 중심 패치들
    if not positions:
        return None

    ds, cs, ms = [], [], []
    for cx, cy in positions:
        patch = extract_window(feat, cx, cy, win_w, win_h)
        d, c, m = kbrs_scores_from_patch(patch, threshold=threshold)
        ds.append(d)
        cs.append(c)
        ms.append(m)

    if not ds:
        return None

    ds_arr = np.array(ds, dtype=np.float32)
    cs_arr = np.array(cs, dtype=np.float32)
    ms_arr = np.array(ms, dtype=np.float32)

    return {
        "replay_id": replay_id,
        "image_id": img_id,
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


def load_coco_gt(label_root: str, replay_id: str, label_method: str) -> COCO:
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
    pred_dir = os.path.join(
        pred_root, model_name, f"model_{epoch:03d}", f"{replay_id}.rep"
    )
    pred_path = os.path.join(pred_dir, f"{label_method}.json")
    if not os.path.isfile(pred_path):
        print(f"[!] Warning: Prediction file not found: {pred_path} -> Proceeding with fallback/empty detections.")
        return {}

    import json

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


def compute_kbrs_for_replay(
    replay_id: str,
    mode: str,
    coco_gt: COCO,
    args: argparse.Namespace,
    preds_by_img: Optional[Dict[int, List[dict]]] = None,
    model_tag: Optional[str] = None,
) -> Tuple[float, float, float, int]:
    """
    KBRS metric score (mean_density / mean_centeredness / mean_mixture)를
    replay 단위로 계산해서 반환.
    """
    window_val = getattr(args, "window", "16,10")
    if isinstance(window_val, str):
        win_w, win_h = map(int, window_val.split(","))
    else:
        win_w, win_h = map(int, window_val)

    stride_val = getattr(args, "stride", "1,1")
    if isinstance(stride_val, str):
        stride_x, stride_y = map(int, stride_val.split(","))
    else:
        stride_x, stride_y = map(int, stride_val)

    feature_ext = getattr(args, "feature_ext", ".npy")
    use_file_name = getattr(args, "use_file_name", False)
    threshold = getattr(args, "threshold", 0.25)
    num_workers = getattr(args, "num_workers", 1)
    max_frames = getattr(args, "max_frames", -1)

    images = list(coco_gt.dataset.get("images", []))
    if max_frames > 0:
        images = images[:max_frames]

    tasks: List[Tuple] = []

    for img in images:
        img_id = int(img["id"])
        file_name = img.get("file_name", "")
        img_w = int(img.get("width", 0))
        img_h = int(img.get("height", 0))

        if mode == "feature":
            positions = None
            source_tag = "feature"

        elif mode == "gt":
            ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
            anns = coco_gt.loadAnns(ann_ids) if ann_ids else []
            if not anns:
                continue
            positions = [centroid_from_coco_ann(ann, img_w, img_h) for ann in anns]
            if not positions:
                continue
            source_tag = "gt"

        else:  # mode == "model"
            assert preds_by_img is not None
            img_preds = preds_by_img.get(img_id, [])
            if not img_preds:
                continue
            positions = [
                centroid_from_coco_ann(det, img_w, img_h)
                for det in img_preds
                if "bbox" in det and det["bbox"] is not None
            ]
            if not positions:
                continue
            source_tag = f"model:{model_tag}"

        tasks.append(
            (
                mode,
                replay_id,
                img_id,
                file_name,
                img_w,
                img_h,
                args.input_root,
                feature_ext,
                use_file_name,
                win_w,
                win_h,
                stride_x,
                stride_y,
                threshold,
                positions,
                source_tag,
            )
        )

    if not tasks:
        print(f"[KBRS] No tasks for replay={replay_id}.")
        return float("nan"), float("nan"), float("nan"), 0

    rows: List[Dict] = []

    if args.num_workers == 1:
        for idx, t in enumerate(tasks, 1):
            row = _process_image_worker(t)
            if row is not None:
                rows.append(row)
            if idx % 200 == 0:
                print(
                    f"[KBRS replay={replay_id}] processed {idx}/{len(tasks)} images (mode={mode})"
                )
    else:
        n_workers = args.num_workers or cpu_count()
        print(
            f"[KBRS] Using {n_workers} worker processes for replay={replay_id} "
            f"(tasks={len(tasks)})"
        )
        with Pool(processes=n_workers) as pool:
            for idx, row in enumerate(
                pool.imap_unordered(_process_image_worker, tasks), 1
            ):
                if row is not None:
                    rows.append(row)
                if idx % 200 == 0:
                    print(
                        f"[KBRS replay={replay_id}] processed {idx}/{len(tasks)} images (mode={mode})"
                    )

    if not rows:
        print(f"[KBRS] No rows for replay={replay_id}.")
        return float("nan"), float("nan"), float("nan"), 0

    df = pd.DataFrame(rows)
    mean_density = float(df["mean_density"].mean())
    mean_centered = float(df["mean_centeredness"].mean())
    mean_mixture = float(df["mean_mixture"].mean())
    num_images = int(df["image_id"].nunique())

    print(
        f"[KBRS replay={replay_id}] "
        f"num_images={num_images}, "
        f"mean_density={mean_density:.4f}, "
        f"mean_centeredness={mean_centered:.4f}, "
        f"mean_mixture={mean_mixture:.4f}"
    )

    return mean_density, mean_centered, mean_mixture, num_images


# =====================================================================
# 2. IC metric (intersection ratio, ic@000/ic_multi/ic_ratio/median_ir/p90_ir)
#    → 기존 evaluator(eval_kernel_from_coco)만 사용
# =====================================================================

def compute_ic_for_replay(
    replay_id: str,
    mode: str,
    coco_gt: COCO,
    args: argparse.Namespace,
    preds_all: Optional[List[dict]] = None,
    model_tag: Optional[str] = None,
    skip_missing_preds: bool = False,
) -> Dict[str, float]:
    """
    IC metric을 replay 단위로 계산.
    skip_missing_preds가 True이면 예측이 존재하는 프레임만 조건부 평가합니다.
    """
    if isinstance(args.ic_kernel, str):
        ic_x_len, ic_y_len = map(int, args.ic_kernel.split(","))
    else:
        ic_x_len, ic_y_len = map(int, args.ic_kernel)

    if isinstance(args.ic_grid, str):
        ic_grid_w, ic_grid_h = map(int, args.ic_grid.split(","))
    else:
        ic_grid_w, ic_grid_h = map(int, args.ic_grid)

    if isinstance(args.ic_maxcoord, str):
        ic_max_x, ic_max_y = map(float, args.ic_maxcoord.split(","))
    else:
        ic_max_x, ic_max_y = map(float, args.ic_maxcoord)

    images = list(coco_gt.dataset.get("images", []))
    max_frames = getattr(args, "max_frames", -1)
    if max_frames > 0:
        images = images[:max_frames]

    num_images = len(images)
    num_preds = len(preds_all) if preds_all else 0

    print(
        f"[IC replay={replay_id}] start "
        f"(mode={mode}, kernel={ic_x_len}x{ic_y_len}, "
        f"grid={ic_grid_w}x{ic_grid_h}, maxcoord=({ic_max_x},{ic_max_y}), "
        f"num_images={num_images}, num_preds={num_preds}, skip_missing={skip_missing_preds})"
    )

    # ------------------------------------------------------------
    # mode=gt: GT annotations를 "preds_all" 형태(list[dict])로 만들어
    #         eval_kernel_from_coco를 호출해서 IC를 계산한다.
    # ------------------------------------------------------------
    if mode == "gt":
        gt_dets: List[dict] = []
        for img in images:
            img_id = int(img["id"])
            ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
            anns = coco_gt.loadAnns(ann_ids) if ann_ids else []
            for ann in anns:
                bbox = ann.get("bbox", None)
                if bbox is None:
                    continue
                gt_dets.append(
                    {
                        "image_id": img_id,
                        "bbox": bbox,
                        "score": 1.0,  # GT는 score가 없으니 1로 채움
                        "category_id": int(ann.get("category_id", 1)),
                    }
                )

        if gt_dets:
            tag = f"gt_r{replay_id}"
            print(
                f"[IC replay={replay_id}] calling eval_kernel_from_coco(...) for GT (n={len(gt_dets)})"
            )
            row, per_img_ic, agg_ic = eval_kernel_from_coco(
                coco_gt,
                gt_dets,
                name=tag,
                kernel=(ic_x_len, ic_y_len),
                grid=(ic_grid_w, ic_grid_h),
                maxcoord=(ic_max_x, ic_max_y),
                skip_missing_preds=skip_missing_preds,
            )

            mean_ir = agg_ic.get("mean_ir", float("nan"))
            median_ir = agg_ic.get("median_ir", float("nan"))
            coverage_any = agg_ic.get("coverage_any", float("nan"))
            multi_cov = agg_ic.get("multi_coverage", float("nan"))
            n_img_ic = agg_ic.get("num_images", len(per_img_ic))

            print(
                f"[IC replay={replay_id}] finished GT eval_kernel_from_coco: "
                f"num_images={n_img_ic}, "
                f"mean_ir={mean_ir:.4f}, median_ir={median_ir:.4f}, "
                f"coverage_any={coverage_any:.4f}, multi_coverage={multi_cov:.4f}"
            )
            print(f"[IC replay={replay_id}] GT summary row: {row}")
            return row

        print(f"[IC replay={replay_id}] GT has no bbox annotations → NaN filled.")
        return {
            "kernel": f"{ic_x_len}x{ic_y_len}",
            "num_images": num_images,
            "ic@000": float("nan"),
            "ic@030": float("nan"),
            "ic@050": float("nan"),
            "ic_multi": float("nan"),
            "ic_ratio": float("nan"),
            "median_ir": float("nan"),
            "p90_ir": float("nan"),
        }


    if mode == "model" and preds_all:
        # eval_kernel_from_coco 안에서 coco → labels_tests → eval_intersection_run 이 수행됨
        print(f"[IC replay={replay_id}] calling eval_kernel_from_coco(...)")
        row, per_img_ic, agg_ic = eval_kernel_from_coco(
            coco_gt,
            preds_all,
            name=f"{model_tag}_r{replay_id}",
            kernel=(ic_x_len, ic_y_len),
            grid=(ic_grid_w, ic_grid_h),
            maxcoord=(ic_max_x, ic_max_y),
            skip_missing_preds=skip_missing_preds,
        )

        # agg_ic에는 mean_ir / median_ir / coverage_any / multi_coverage / num_images 등이 들어 있음
        mean_ir = agg_ic.get("mean_ir", float("nan"))
        median_ir = agg_ic.get("median_ir", float("nan"))
        coverage_any = agg_ic.get("coverage_any", float("nan"))
        multi_cov = agg_ic.get("multi_coverage", float("nan"))
        n_img_ic = agg_ic.get("num_images", len(per_img_ic))

        print(
            f"[IC replay={replay_id}] finished eval_kernel_from_coco: "
            f"num_images={n_img_ic}, "
            f"mean_ir={mean_ir:.4f}, median_ir={median_ir:.4f}, "
            f"coverage_any={coverage_any:.4f}, multi_coverage={multi_cov:.4f}"
        )
        print(f"[IC replay={replay_id}] summary row: {row}")

        return row

    # feature/gt 모드이거나, preds_all 이 없는 경우 → 계산 스킵, NaN 로 채움
    print(
        f"[IC replay={replay_id}] skip IC calculation "
        f"(mode={mode}, num_preds={num_preds}) → NaN filled."
    )
    return {
        "kernel": f"{ic_x_len}x{ic_y_len}",
        "num_images": num_images,
        "ic@000": float("nan"),
        "ic@030": float("nan"),
        "ic@050": float("nan"),
        "ic_multi": float("nan"),
        "ic_ratio": float("nan"),
        "median_ir": float("nan"),
        "p90_ir": float("nan"),
    }



def compute_multi_region_for_replay(
    replay_id: str,
    coco_gt: COCO,
    preds_by_img: Optional[Dict[int, List[dict]]],
    grid_size: Tuple[int, int] = (128, 128),
) -> Dict[str, float]:
    """
    Computes Multi-Region Finding metrics for a replay sequence:
      - cwo: Consensus-Weighted Overlap
      - m_cti: Multi-Track Camera Thrashing Index
      - jerk: Jerk penalty
      - jump_rate: Teleport jump rate
      - event_recall: Objective Event Recall (R_event)
      - pairwise_overlap: Viewport Redundancy (IoU among viewports)
    """
    images = list(coco_gt.dataset.get("images", []))
    if not images:
        return {
            "cwo": float("nan"),
            "m_cti": float("nan"),
            "jerk": float("nan"),
            "jump_rate": float("nan"),
            "event_recall": float("nan"),
            "pairwise_overlap": float("nan"),
        }

    evaluator = MultiRegionEvaluator(
        grid_size=grid_size,
        jump_threshold=35.0,
        box_format="xyxy",
        m_cti_weights=(0.4, 0.4, 0.2),
    )

    viewports_seq = []
    consensus_maps_seq = []
    event_coords_seq = []

    grid_w, grid_h = grid_size

    for img in images:
        img_id = int(img["id"])
        h_img = float(img.get("height", grid_h))
        w_img = float(img.get("width", grid_w))

        img_boxes = []
        if preds_by_img and img_id in preds_by_img:
            for det in preds_by_img[img_id]:
                b = det.get("bbox", None)
                if b is not None and len(b) == 4:
                    x, y, w, h = b
                    x1 = (x / w_img) * grid_w
                    y1 = (y / h_img) * grid_h
                    x2 = ((x + w) / w_img) * grid_w
                    y2 = ((y + h) / h_img) * grid_h
                    img_boxes.append([x1, y1, x2, y2])

        if not img_boxes:
            img_boxes = [[0.0, 0.0, 32.0, 32.0]]
        viewports_seq.append(np.array(img_boxes, dtype=np.float32))

        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids) if ann_ids else []

        c_map = np.zeros((grid_h, grid_w), dtype=np.float32)
        events = []
        for ann in anns:
            bbox = ann.get("bbox", None)
            if bbox is not None and len(bbox) == 4:
                x, y, w, h = bbox
                cx = (x + w / 2.0) / w_img * grid_w
                cy = (y + h / 2.0) / h_img * grid_h
                events.append([cx, cy])

                x1_i = int(np.clip((x / w_img) * grid_w, 0, grid_w))
                x2_i = int(np.clip(((x + w) / w_img) * grid_w, 0, grid_w))
                y1_i = int(np.clip((y / h_img) * grid_h, 0, grid_h))
                y2_i = int(np.clip(((y + h) / h_img) * grid_h, 0, grid_h))
                if x2_i > x1_i and y2_i > y1_i:
                    c_map[y1_i:y2_i, x1_i:x2_i] += 1.0

        if c_map.max() > 0:
            c_map /= c_map.max()

        consensus_maps_seq.append(c_map)
        event_coords_seq.append(np.array(events, dtype=np.float32) if events else np.empty((0, 2), dtype=np.float32))

    metrics = evaluator.evaluate_sequence(
        viewports_seq=viewports_seq,
        consensus_maps_seq=consensus_maps_seq,
        event_coords_seq=event_coords_seq,
    )
    return metrics


# =====================================================================
# 3. CLI / main: Single-Region & Multi-Region Finding Evaluator
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Single-Region Finding (KBRS, COCO IC) & "
            "Multi-Region Finding (CWO, M-CTI, Event Recall, Pairwise Overlap) Evaluator."
        )
    )
    p.add_argument("--replays", type=str, nargs="+", required=True)
    p.add_argument("--input-root", default="/workspace/data/input/dst")
    p.add_argument("--label-root", default="/workspace/data/label/dst")
    p.add_argument("--label-method", default="all_correct")
    p.add_argument(
        "--mode",
        choices=["feature", "gt", "model"],
        default="feature",
    )
    p.add_argument(
        "--task",
        choices=["single", "multi", "all"],
        default="all",
        help="Evaluation task: single (Single-Region Finding), multi (Multi-Region Finding), or all.",
    )
    p.add_argument("--pred-root", default="/workspace/predictions")
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--epoch", type=int, default=None)

    # KBRS-related
    p.add_argument("--window", default="20,12")     # KBRS kernel (w,h)
    p.add_argument("--stride", default="1,1")      # KBRS stride (feature 모드용)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--use-file-name", action="store_true")
    p.add_argument("--feature-ext", default=".npy")
    p.add_argument(
        "--skip-kbrs",
        action="store_true",
        help="KBRS density/centeredness/mixture 계산을 건너뛰고 NaN으로 채움.",
    )

    p.add_argument(
        "--csv-out",
        default=None,
        help="최종 replay-level summary CSV 경로 (default: /workspace/results/{mode or model_name}.csv)",
    )

    # IC metric kernel 관련 (기존 evaluator와 동일 파라미터)
    p.add_argument("--ic-kernel", default="20,12")
    p.add_argument("--ic-grid", default="128,128")
    p.add_argument("--ic-maxcoord", default="3456,3720")

    p.add_argument("--num-workers", type=int, default=0)

    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "model":
        if not args.model_name or args.epoch is None:
            raise ValueError(
                "mode=model 인 경우 --model-name 과 --epoch 를 반드시 지정해야 합니다."
            )

    # CSV 경로 기본값
    if not args.csv_out:
        base_root = "/workspace/results"
        os.makedirs(base_root, exist_ok=True)
        if args.mode == "model" and args.model_name:
            csv_name = f"{args.model_name}_e{args.epoch}.csv"
        else:
            csv_name = f"{args.mode}.csv"
        args.csv_out = os.path.join(base_root, csv_name)

    print(f"[INFO] Output CSV (replay-level rows): {args.csv_out}")

    replay_rows: List[Dict] = []

    for replay_id in args.replays:
        replay_id = str(replay_id)
        print(f"\n[Replay] {replay_id}")

        coco_gt = load_coco_gt(
            label_root=args.label_root,
            replay_id=replay_id,
            label_method=args.label_method,
        )

        images = list(coco_gt.dataset.get("images", []))
        if args.max_frames > 0:
            images = images[: args.max_frames]

        preds_by_img: Optional[Dict[int, List[dict]]] = None
        preds_all: List[dict] = []
        model_tag: Optional[str] = None

        if args.mode == "model":
            preds_by_img = load_coco_preds(
                pred_root=args.pred_root,
                model_name=args.model_name,
                epoch=args.epoch,
                replay_id=replay_id,
                label_method=args.label_method,
            )
            for dets in preds_by_img.values():
                preds_all.extend(dets)
            model_tag = f"{args.model_name}_e{args.epoch}"

        # 1) Single-Region Finding Metrics (KBRS + IC)
        if args.task in ["single", "all"]:
            if args.skip_kbrs:
                mean_density = float("nan")
                mean_centered = float("nan")
                mean_mixture = float("nan")
                num_images_kbrs = len(images)
                print(f"[Single-Region replay={replay_id}] KBRS skipped (--skip-kbrs)")
            else:
                mean_density, mean_centered, mean_mixture, num_images_kbrs = compute_kbrs_for_replay(
                    replay_id=replay_id,
                    mode=args.mode,
                    coco_gt=coco_gt,
                    args=args,
                    preds_by_img=preds_by_img,
                    model_tag=model_tag,
                )

            ic_row = compute_ic_for_replay(
                replay_id=replay_id,
                mode=args.mode,
                coco_gt=coco_gt,
                args=args,
                preds_all=preds_all if preds_all else None,
                model_tag=model_tag,
            )
        else:
            mean_density = float("nan")
            mean_centered = float("nan")
            mean_mixture = float("nan")
            ic_row = {"kernel": "20x12", "num_images": len(images)}

        # 2) Multi-Region Finding Metrics (CWO, M-CTI, Event Recall, Pairwise Overlap)
        if args.task in ["multi", "all"]:
            multi_row = compute_multi_region_for_replay(
                replay_id=replay_id,
                coco_gt=coco_gt,
                preds_by_img=preds_by_img,
            )
        else:
            multi_row = {
                "cwo": float("nan"),
                "m_cti": float("nan"),
                "jerk": float("nan"),
                "jump_rate": float("nan"),
                "event_recall": float("nan"),
                "pairwise_overlap": float("nan"),
            }

        # 3) Merge replay-level summary row
        final_row = dict(ic_row)
        final_row.update(multi_row)
        final_row.update({
            "replay": replay_id,
            "kernel": ic_row.get("kernel"),
            "num_images": ic_row.get("num_images", len(images)),
            "mean_density": mean_density,
            "mean_centeredness": mean_centered,
            "mean_mixture": mean_mixture,
            "mode": args.mode,
            "task": args.task,
            "model_name": args.model_name if args.mode == "model" else None,
            "epoch": args.epoch if args.mode == "model" else None,
        })
        replay_rows.append(final_row)

    if not replay_rows:
        print("[Info] No replay rows to write CSV.")
        return

    df_all = pd.DataFrame(replay_rows).sort_values("replay")
    out_dir = os.path.dirname(args.csv_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_all.to_csv(args.csv_out, index=False)
    print(f"\n[Info] saved replay-level summary to: {args.csv_out}")
    print(df_all)


if __name__ == "__main__":
    main()
