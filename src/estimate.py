# src/kbrs_from_input.py
from __future__ import annotations

import os
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO

import config
from config import KBRS_PARAMS
from model.utils import normalize_projections
from model.kbrs import KBRSConvScorer

from utils.logger import Logger


# ----------------------------------------------------------------------
# COCO / feature 경로 관련 유틸
# ----------------------------------------------------------------------
def build_feature_path_from_image(
    input_root: str,
    replay_id: str,
    img: dict,
    *,
    use_file_name: bool = True,
    feature_ext: str = ".npy",
) -> str:
    """
    COCO image 엔트리 하나(img)로부터 feature .npy 경로를 구성.

    기본 가정:
      - feature 디렉터리: {input_root}/{replay_id}.rep/
      - 파일명 매핑:
          use_file_name=True  → file_name에서 확장자만 교체
          use_file_name=False → image id를 문자열로 써서 {id}.npy
    """
    rep_dir = os.path.join(input_root, f"{replay_id}.rep")

    if use_file_name:
        file_name: str = img["file_name"]
        base, _ext = os.path.splitext(file_name)
        fname = base + feature_ext
    else:
        img_id = img["id"]
        fname = f"{img_id}{feature_ext}"

    return os.path.join(rep_dir, fname)


def iter_frames_from_coco(
    label_root: str,
    replay_id: str,
    *,
    label_method: str,
) -> List[dict]:
    """
    {label_root}/{replay_id}.rep/{label_method}.json 에서 COCO 객체를 읽고,
    images 리스트를 그대로 반환.
    """
    gt_dir = os.path.join(label_root, f"{replay_id}.rep")
    gt_path = os.path.join(gt_dir, f"{label_method}.json")

    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Ground truth (COCO) file not found: {gt_path}")

    coco = COCO(gt_path)
    images = list(coco.dataset.get("images", []))
    return images


# ----------------------------------------------------------------------
# KBRS map 계산: torch backend (모델과 동일 정의)
# ----------------------------------------------------------------------
def compute_kbrs_map_torch(
    x: np.ndarray,
    *,
    device: torch.device,
    kbrs_params: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    KBRSConvScorer 를 그대로 사용해서 density / centeredness / mixture 맵을 계산.

    feat: (C,H,W) numpy array 또는 torch.Tensor
    """
    # 1) numpy → torch 변환 + 배치 차원 추가
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        x = x
    else:
        raise TypeError(f"Unsupported feat type: {type(x)}")

    # (C,H,W) → (1,C,H,W), 이미 4D면 그대로 사용
    if x.ndim == 3:
        x = x.unsqueeze(0)
    elif x.ndim == 4:
        pass
    else:
        raise ValueError(f"Expected 3D or 4D feat, got shape {tuple(x.shape)}")

    x = x.to(device=device, dtype=torch.float32)

    # 2) KBRS 설정 (config.KBRS_PARAMS 기반)
    if kbrs_params is None:
        kbrs_params = getattr(config, "KBRS_PARAMS", {})

    region_size = tuple(kbrs_params.get("region_size", (20, 12)))
    weights = kbrs_params.get(
        "score_weights",
        kbrs_params.get(
            "weights", {"density": 1.0, "mixture": 1.0, "centeredness": 1.0}
        ),
    )
    projections = kbrs_params.get("projections", None)
    mixture_tau = kbrs_params.get("mixture_tau", 2.0)
    mixture_mode = kbrs_params.get("mixture_mode", "confusion")
    mixture_power = kbrs_params.get("mixture_power", 1.0)
    mask_channel = kbrs_params.get("mask_channel", None)
    mask_gain = kbrs_params.get("mask_gain", 1.0)
    score_stride = kbrs_params.get("score_stride", 1)
    downsample_before = kbrs_params.get("downsample_before", None)

    # 필요하다면 여기서 normalize_projections(...) 호출해서 projections 정규화해도 됨
    scorer = KBRSConvScorer(
        region_size=region_size,
        weights=weights,
        projections=projections,
        mixture_tau=mixture_tau,
        mixture_mode=mixture_mode,
        mixture_power=mixture_power,
        mask_channel=mask_channel,
        mask_gain=mask_gain,
        score_stride=score_stride,
        downsample_before=downsample_before,
    ).to(device=device, dtype=torch.float32)

    scorer.eval()
    with torch.no_grad():
        score_map, comp_maps = scorer(x)

    def _get(name: str) -> np.ndarray:
        if name in comp_maps:
            return comp_maps[name].squeeze(0).detach().cpu().numpy()
        # 해당 컴포넌트가 없으면 score_map 모양에 맞는 0으로 채움
        return np.zeros_like(score_map.squeeze(0).detach().cpu().numpy())

    density_map = _get("density")
    centeredness_map = _get("centeredness")
    mixture_map = _get("mixture")

    return density_map, centeredness_map, mixture_map


# ----------------------------------------------------------------------
# KBRS map 계산: numpy backend (근사 버전)
# ----------------------------------------------------------------------
def compute_kbrs_map_numpy(
    feat: np.ndarray,
    *,
    win_w: int,
    win_h: int,
    stride_x: int = 1,
    stride_y: int = 1,
    threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    순수 numpy 로 근사 KBRS-like density/centeredness/mixture 맵을 계산.
    replay 단위 트렌드용.
    """
    if feat.ndim == 3:
        C, H, W = feat.shape
        summed = feat.sum(axis=0)  # (H,W)
    elif feat.ndim == 2:
        H, W = feat.shape
        summed = feat
    else:
        raise ValueError(f"Unexpected feature shape: {feat.shape}")

    binary = (summed > threshold).astype(np.float32)

    out_h = 1 + (H - win_h) // stride_y if H >= win_h else 0
    out_w = 1 + (W - win_w) // stride_x if W >= win_w else 0

    density_map = np.zeros((out_h, out_w), dtype=np.float32)
    centeredness_map = np.zeros_like(density_map)
    mixture_map = np.zeros_like(density_map)

    for oy in range(out_h):
        for ox in range(out_w):
            y0 = oy * stride_y
            x0 = ox * stride_x
            y1 = y0 + win_h
            x1 = x0 + win_w

            patch = binary[y0:y1, x0:x1]
            area = float(win_w * win_h)
            mass = patch.sum()

            if area <= 0:
                continue

            density = float(mass / area)

            if mass <= 0:
                centered = 0.0
            else:
                ys, xs = np.nonzero(patch)
                tcx = xs.mean()
                tcy = ys.mean()
                cx = (win_w - 1) / 2.0
                cy = (win_h - 1) / 2.0
                dist = float(np.hypot(tcx - cx, tcy - cy))
                half_diag = float(np.hypot(win_w, win_h) / 2.0)
                centered = max(0.0, 1.0 - dist / (half_diag + 1e-12))

            mixture = density * centered

            density_map[oy, ox] = density
            centeredness_map[oy, ox] = centered
            mixture_map[oy, ox] = mixture

    return density_map, centeredness_map, mixture_map


# ----------------------------------------------------------------------
# Argparse
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute KBRS-like density/centeredness/mixture scores over replays, "
            "aggregated per replay_id."
        )
    )
    p.add_argument(
        "--replays",
        type=str,
        nargs="+",
        required=True,
        help="리플레이 ID 목록 (예: 275 3613 4520)",
    )
    p.add_argument(
        "--input-root",
        default="/workspace/data/input/dst",
        help="입력 feature .npy 들이 있는 루트 디렉터리",
    )
    p.add_argument(
        "--label-root",
        default="/workspace/data/label/dst",
        help="COCO ground_truth 가 있는 루트 디렉터리",
    )
    p.add_argument(
        "--label-method",
        default="all_correct",
        help="사용할 COCO GT 파일 이름 (예: all_correct → {replay_id}.rep/all_correct.json)",
    )
    p.add_argument(
        "--window",
        default="20,12",
        help="(numpy backend용) 커널 크기 (w,h). 예: '20,12'",
    )
    p.add_argument(
        "--stride",
        default="1,1",
        help="(numpy backend용) stride (sx,sy). 예: '1,1'",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="numpy backend에서 binary map threshold (default: 0.0)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="0보다 크면 각 replay에서 images 리스트 중 앞에서 이 개수만 사용.",
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
        "--model-name",
        default=None,
        help="이 run 을 구분할 model/tag 이름. csv 기본 파일명에 사용.",
    )
    p.add_argument(
        "--backend",
        choices=["torch", "numpy"],
        default="torch",
        help="KBRS 계산 backend: 'torch'(KBRSConvScorer) / 'numpy'(근사).",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="torch backend에서 device 선택. auto=CUDA 있으면 cuda, 아니면 cpu.",
    )
    p.add_argument(
        "--csv-out",
        default=None,
        help=(
            "결과를 저장할 CSV 경로. "
            "지정하지 않으면 /workspace/results/{model_name or kbrs_{backend}}.csv"
        ),
    )

    return p.parse_args()


# ----------------------------------------------------------------------
# 메인 로직 (replay_id 단위 집계)
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    win_w, win_h = map(int, args.window.split(","))
    stride_x, stride_y = map(int, args.stride.split(","))

    # backend/device 설정
    if args.backend == "torch":
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif args.device == "cuda":
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        Logger.info(f"[KBRS] Using torch backend on device={device}")
    else:
        device = torch.device("cpu")
        Logger.info("[KBRS] Using numpy backend (CPU only)")

    # csv 기본 경로 설정 (모델당 CSV 하나)
    if not args.csv_out:
        base_root = "/workspace/results"
        os.makedirs(base_root, exist_ok=True)

        if args.model_name:
            csv_name = f"{args.model_name}.csv"
        else:
            csv_name = f"kbrs_{args.backend}.csv"

        args.csv_out = os.path.join(base_root, csv_name)

    Logger.info(f"[KBRS] Output CSV (per replay row): {args.csv_out}")

    all_rows: List[Dict[str, Any]] = []

    # 리플레이 루프
    for replay_id in args.replays:
        replay_id_str = str(replay_id)
        Logger.info(f"[Replay] replay_id={replay_id_str}")

        try:
            images = iter_frames_from_coco(
                label_root=args.label_root,
                replay_id=replay_id_str,
                label_method=args.label_method,
            )
        except FileNotFoundError as e:
            Logger.error(str(e))
            continue

        if args.max_frames > 0:
            images = images[: args.max_frames]

        Logger.info(f"[Replay {replay_id_str}] num_images={len(images)}")

        # 이 replay 안에서 per-image mean 들을 모아서, 끝에서 replay 단위 평균
        image_means_density: List[float] = []
        image_means_centered: List[float] = []
        image_means_mixture: List[float] = []

        for idx, img in enumerate(images):
            img_id = img["id"]
            file_name = img.get("file_name", "")
 
            feat_path = build_feature_path_from_image(
                input_root=args.input_root,
                replay_id=replay_id_str,
                img=img,
                use_file_name=args.use_file_name,
                feature_ext=args.feature_ext,
            )

            if not os.path.isfile(feat_path):
                Logger.warning(
                    f"[Warn] feature not found for replay={replay_id_str}, "
                    f"image_id={img_id}, file_name='{file_name}': {feat_path}"
                )
                continue

            feat = np.load(feat_path)  # (C,H,W) 또는 (H,W)

            if args.backend == "torch":
                density_map, centeredness_map, mixture_map = compute_kbrs_map_torch(
                    feat,
                    device=device,
                    kbrs_params=getattr(config, "KBRS_PARAMS", None),
                )
            else:
                density_map, centeredness_map, mixture_map = compute_kbrs_map_numpy(
                    feat,
                    win_w=win_w,
                    win_h=win_h,
                    stride_x=stride_x,
                    stride_y=stride_y,
                    threshold=args.threshold,
                )

            image_means_density.append(float(density_map.mean()))
            image_means_centered.append(float(centeredness_map.mean()))
            image_means_mixture.append(float(mixture_map.mean()))

            if idx % 200 == 0:
                Logger.info(
                    f"[Replay {replay_id_str} Image {idx}/{len(images)} id={img_id}] "
                    f"mean_density={image_means_density[-1]:.4f}, "
                    f"mean_centered={image_means_centered[-1]:.4f}, "
                    f"mean_mixture={image_means_mixture[-1]:.4f}"
                )

        if not image_means_density:
            Logger.info(
                f"[Replay {replay_id_str}] No images with features processed, skipping."
            )
            continue

        # replay 단위 평균 (image mean 들의 평균)
        mean_density_replay = float(np.mean(image_means_density))
        mean_centered_replay = float(np.mean(image_means_centered))
        mean_mixture_replay = float(np.mean(image_means_mixture))

        row = {
            "replay_id": replay_id_str,
            "backend": args.backend,
            "model_name": args.model_name,
            "num_images": len(image_means_density),
            "mean_density": mean_density_replay,
            "mean_centeredness": mean_centered_replay,
            "mean_mixture": mean_mixture_replay,
        }
        all_rows.append(row)

        Logger.info(
            f"[Replay {replay_id_str} summary] "
            f"num_images={row['num_images']}, "
            f"mean_density={mean_density_replay:.4f}, "
            f"mean_centeredness={mean_centered_replay:.4f}, "
            f"mean_mixture={mean_mixture_replay:.4f}"
        )

    if not all_rows:
        Logger.info("[KBRS] No replay had valid images; not writing CSV.")
        return

    df = pd.DataFrame(all_rows).sort_values("replay_id")

    out_dir = os.path.dirname(args.csv_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(args.csv_out, index=False)
    Logger.info(f"[KBRS] Saved replay-level KBRS summary to: {args.csv_out}")
    Logger.info(
        f"[KBRS] Global mean(mean_density)={df['mean_density'].mean():.4f}, "
        f"mean(mean_centeredness)={df['mean_centeredness'].mean():.4f}, "
        f"mean(mean_mixture)={df['mean_mixture'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
