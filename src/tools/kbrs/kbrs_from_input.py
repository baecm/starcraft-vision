from __future__ import annotations

import os
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from multiprocessing import Pool, cpu_count


# ---------------------------------------------------------------------
# KBRS 유틸: 가우시안 커널 / 패치 스코어 / 슬라이딩 윈도우
# ---------------------------------------------------------------------
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
    patch: (C,H,W) 또는 (H,W)

    - density: binary map (patch2d > threshold)의 평균
    - centeredness: binary map에 2D gaussian weight를 곱한 weighted mean
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
        # 빈 패치 리턴
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
    전체 feature 맵 위를 슬라이딩 윈도우로 훑으면서 KBRS map 3개(density/centered/mixture)를 만든다.
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


# ---------------------------------------------------------------------
# COCO / 경로 유틸
# ---------------------------------------------------------------------
def centroid_from_coco_ann(ann: dict, img_w: int, img_h: int) -> Tuple[float, float]:
    """
    COCO annotation(예측/GT 둘 다)에 대해 bbox 중심을 픽셀 좌표로 반환.
    segmentation 이 있어도 여기서는 bbox만 사용.
    """
    if "bbox" in ann and ann["bbox"] is not None:
        x, y, w, h = ann["bbox"]
        return float(x + w / 2.0), float(y + h / 2.0)
    # fallback: 이미지 중앙
    return float(img_w) / 2.0, float(img_h) / 2.0


def build_feature_path_from_meta(
    input_root: str,
    replay_id: str,
    file_name: str,
    img_id: int,
    use_file_name: bool,
    feature_ext: str,
) -> str:
    """
    feature 경로:
      {input_root}/{replay_id}.rep/{file_name or id}.npy
    """
    rep_dir = os.path.join(input_root, f"{replay_id}.rep")

    if use_file_name and file_name:
        base, _ext = os.path.splitext(file_name)
        fname = base + feature_ext
    else:
        fname = f"{img_id}{feature_ext}"

    return os.path.join(rep_dir, fname)


def load_coco_gt(label_root: str, replay_id: str, label_method: str) -> COCO:
    """
    GT COCO 경로:
      {label_root}/{replay_id}.rep/{label_method}.json
    예: /workspace/data/label/dst/275.rep/all_correct.json
    """
    gt_dir = os.path.join(label_root, f"{replay_id}.rep")
    gt_path = os.path.join(gt_dir, f"{label_method}.json")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Ground truth (COCO) file not found: {gt_path}")
    coco = COCO(gt_path)
    return coco


def load_coco_preds(
    pred_root: str,
    model_name: str,
    epoch: int,
    replay_id: str,
    label_method: str,
) -> Dict[int, List[dict]]:
    """
    모델 prediction COCO 경로:
      {pred_root}/{model_name}/model_{epoch}/{replay_id}.rep/{label_method}.json
    """
    pred_dir = os.path.join(
        pred_root,
        model_name,
        f"model_{epoch}",
        f"{replay_id}.rep",
    )
    pred_path = os.path.join(pred_dir, f"{label_method}.json")
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    import json

    with open(pred_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    # COCO detection list 로 가정
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


# ---------------------------------------------------------------------
# 멀티프로세싱 워커
# ---------------------------------------------------------------------
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
        source_tag,  # "feature", "gt", "model:xxx"
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

    # -------------------
    # mode=feature
    # -------------------
    if mode == "feature":
        density_map, centeredness_map, mixture_map = compute_kbrs_grid(
            feat,
            win_w=win_w,
            win_h=win_h,
            stride_x=stride_x,
            stride_y=stride_y,
            threshold=threshold,
        )
        row = {
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
        return row

    # -------------------
    # mode=gt / mode=model
    # -------------------
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

    row = {
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
    return row


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute KBRS-like density/centeredness/mixture scores over input feature maps.\n"
            "Modes:\n"
            "  feature: 슬라이딩 윈도우로 전체 feature 스캔\n"
            "  gt     : GT object 중심 윈도우에서 KBRS 측정\n"
            "  model  : model prediction bbox 중심 윈도우에서 KBRS 측정\n"
        )
    )
    p.add_argument(
        "--replays",
        type=str,
        nargs="+",
        required=True,
        help="리플레이 ID 목록 (예: 275 3613 4664)",
    )
    p.add_argument(
        "--input-root",
        default="/workspace/data/input/dst",
        help="입력 feature .npy 들이 있는 루트 디렉터리 (예: /workspace/data/input/dst)",
    )
    p.add_argument(
        "--label-root",
        default="/workspace/data/label/dst",
        help="COCO ground_truth 가 있는 루트 디렉터리 (예: /workspace/data/label/dst)",
    )
    p.add_argument(
        "--label-method",
        default="all_correct",
        help="라벨링 방법 이름 (파일명으로 사용됨, 예: all_correct → all_correct.json)",
    )
    p.add_argument(
        "--mode",
        choices=["feature", "gt", "model"],
        default="feature",
        help="KBRS 계산 모드: feature(전체 슬라이딩), gt(GT object 중심), model(prediction 중심)",
    )
    p.add_argument(
        "--pred-root",
        default="/workspace/predictions",
        help="모델 prediction JSON 상위 디렉터리 (예: /workspace/predictions)",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="mode=model 일 때 사용할 모델 이름",
    )
    p.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="mode=model 일 때 사용할 epoch 번호 (예: 30 → model_30)",
    )
    p.add_argument(
        "--window",
        default="20,12",
        help="커널 크기 (w,h). 예: '20,12'",
    )
    p.add_argument(
        "--stride",
        default="1,1",
        help="슬라이딩 stride (sx,sy). 예: '4,4' (mode=feature 에서만 사용)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="binary map을 만들 때 사용할 threshold (default: 0.0)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="0보다 크면 images 리스트 중 앞에서 이 개수만 사용.",
    )
    p.add_argument(
        "--use-file-name",
        action="store_true",
        help="feature 파일명을 image['file_name'] 기반으로 만들면 설정. "
        "끄면 image['id'] 기반 {id}.npy 로 가정.",
    )
    p.add_argument(
        "--feature-ext",
        default=".npy",
        help="feature 파일 확장자 (기본 .npy).",
    )
    p.add_argument(
        "--csv-out",
        default=None,
        help=(
            "CSV 출력 경로.\n"
            "  - 디렉터리 경로면: replay별로 kbrs_{mode}_{replay}.csv 저장\n"
            "  - 파일 경로면: 모든 replay 결과를 합쳐 한 파일에 저장"
        ),
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="멀티프로세싱 worker 수 (0이면 cpu_count(), 1이면 단일 프로세스)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------
# 메인 로직
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    win_w, win_h = map(int, args.window.split(","))
    stride_x, stride_y = map(int, args.stride.split(","))

    if args.mode == "model":
        if not args.model_name or args.epoch is None:
            raise ValueError(
                "mode=model 인 경우 --model-name 과 --epoch 를 지정해야 합니다."
            )

    all_rows: List[Dict] = []

    for replay_id in args.replays:
        replay_id = str(replay_id)
        print(f"\n[Replay] {replay_id}")

        # GT COCO 로드 (images 리스트 및 img size 얻기용)
        coco_gt = load_coco_gt(
            label_root=args.label_root,
            replay_id=replay_id,
            label_method=args.label_method,
        )
        images = list(coco_gt.dataset.get("images", []))
        if args.max_frames > 0:
            images = images[: args.max_frames]

        print(f"[Info] replay_id={replay_id}, num_images={len(images)}")

        preds_by_img: Optional[Dict[int, List[dict]]] = None
        model_tag: Optional[str] = None
        if args.mode == "model":
            preds_by_img = load_coco_preds(
                pred_root=args.pred_root,
                model_name=args.model_name,
                epoch=args.epoch,
                replay_id=replay_id,
                label_method=args.label_method,
            )
            model_tag = f"{args.model_name}_e{args.epoch}"

        # ---- per-image task 리스트 구성 ----
        tasks: List[Tuple] = []

        for img in images:
            img_id = int(img["id"])
            file_name = img.get("file_name", "")
            img_w = int(img.get("width", 0))
            img_h = int(img.get("height", 0))

            # mode 별로 positions 준비
            if args.mode == "feature":
                positions = None
                source_tag = "feature"

            elif args.mode == "gt":
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
                    args.mode,
                    replay_id,
                    img_id,
                    file_name,
                    img_w,
                    img_h,
                    args.input_root,
                    args.feature_ext,
                    args.use_file_name,
                    win_w,
                    win_h,
                    stride_x,
                    stride_y,
                    args.threshold,
                    positions,
                    source_tag,
                )
            )

        if not tasks:
            print(f"[Info] No tasks for replay={replay_id}.")
            continue

        rows: List[Dict] = []

        # ---- 멀티프로세싱 실행 ----
        if args.num_workers == 1:
            # 단일 프로세스 (디버깅용)
            for idx, t in enumerate(tasks, 1):
                row = _process_image_worker(t)
                if row is not None:
                    rows.append(row)
                if idx % 200 == 0:
                    print(
                        f"[Replay {replay_id}] processed {idx}/{len(tasks)} images (mode={args.mode})"
                    )
        else:
            n_workers = args.num_workers or cpu_count()
            print(
                f"[Info] Using {n_workers} worker processes for replay={replay_id} (tasks={len(tasks)})"
            )
            with Pool(processes=n_workers) as pool:
                for idx, row in enumerate(
                    pool.imap_unordered(_process_image_worker, tasks), 1
                ):
                    if row is not None:
                        rows.append(row)
                    if idx % 200 == 0:
                        print(
                            f"[Replay {replay_id}] processed {idx}/{len(tasks)} images (mode={args.mode})"
                        )

        if not rows:
            print(f"[Info] No images processed for replay={replay_id}.")
            continue
 
        df = pd.DataFrame(rows).sort_values("image_id")
        print("\n[Summary over images]")
        print(f"  images processed : {len(df)}")
        print(f"  mean(mean_density)      = {df['mean_density'].mean():.4f}")
        print(f"  mean(mean_centeredness) = {df['mean_centeredness'].mean():.4f}")
        print(f"  mean(mean_mixture)      = {df['mean_mixture'].mean():.4f}")

        all_rows.extend(rows)

        # csv-out 이 디렉터리면 replay별 파일로 저장
        if args.csv_out and os.path.isdir(args.csv_out):
            out_path = os.path.join(args.csv_out, f"kbrs_{args.mode}_{replay_id}.csv")
            df.to_csv(out_path, index=False)
            print(f"[Info] saved per-image KBRS summary to: {out_path}")

    # csv-out 이 "파일 경로"면 전체 replay 합쳐서 저장
    if args.csv_out and not os.path.isdir(args.csv_out):
        if not all_rows:
            print("[Info] No data to write CSV.")
        else:
            out_dir = os.path.dirname(args.csv_out)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            df_all = pd.DataFrame(all_rows).sort_values(["replay_id", "image_id"])
            df_all.to_csv(args.csv_out, index=False)
            print(f"[Info] saved combined KBRS summary to: {args.csv_out}")


if __name__ == "__main__":
    main()
