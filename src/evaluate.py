from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import pycocotools.mask as mask_util
from pycocotools.coco import COCO

from detection.coco_utils import get_coco_api_from_dataset
from custom_evaluator import ImageIR, eval_intersection_run
from utils.logger import Logger


# ----------------------------------------------------------------------
# 평가 결과 구조체
# ----------------------------------------------------------------------


@dataclass
class EvalIRResult:
    per_image: List[ImageIR]
    aggregates: Dict[str, float]


# ----------------------------------------------------------------------
# 1) 훈련/검증 루프에서 호출하는 evaluate()
# ----------------------------------------------------------------------


def _encode_mask(mask: np.ndarray) -> dict:
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


@torch.inference_mode()
def _collect_predictions_from_loader(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    score_thresh: float = 0.0,
    max_dets: Optional[int] = None,
    with_masks: bool = True,
) -> List[dict]:
    """
    data_loader에서 모델을 돌려 COCO-style detection 리스트를 생성.
    (훈련용 evaluate()와 CLI에서 공용으로 사용)
    """
    model.eval()
    preds: List[dict] = []
    cpu_device = torch.device("cpu")

    Logger.info("[Eval] Running model on validation set for IR metrics...")
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"].item())
            boxes = output["boxes"].detach().cpu()
            scores = output["scores"].detach().cpu()
            labels = output["labels"].detach().cpu()

            # score 순 정렬
            order = torch.argsort(scores, descending=True)
            boxes = boxes[order]
            scores = scores[order]
            labels = labels[order]

            # score threshold
            if score_thresh > 0.0:
                keep = scores >= score_thresh
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            # max det
            if max_dets is not None and len(scores) > max_dets:
                boxes = boxes[:max_dets]
                scores = scores[:max_dets]
                labels = labels[:max_dets]

            if with_masks and "masks" in output:
                masks = output["masks"].detach().cpu() > 0.5
            else:
                masks = None

            for i in range(len(scores)):
                det: Dict[str, Any] = {
                    "image_id": image_id,
                    "category_id": int(labels[i]),
                    "score": float(scores[i]),
                }
                if masks is not None:
                    m = masks[i, 0].numpy()
                    det["segmentation"] = _encode_mask(m)
                else:
                    box = boxes[i]
                    det["bbox"] = [
                        float(box[0]),
                        float(box[1]),
                        float(box[2] - box[0]),
                        float(box[3] - box[1]),
                    ]
                preds.append(det)

    Logger.info("[Eval] Finished collecting predictions.")
    return preds


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    denom: str = "gt",
    pred_agg: str = "best",
    cat_id: Optional[int] = None,
    window_source: str = "pred",
    window_target: str = "gt",
    win_w: int = 20,
    win_h: int = 12,
) -> EvalIRResult:
    """
    훈련/검증 루프에서 사용하는 Observer 전용 evaluate 함수.

    - GT가 있는 data_loader를 받아서
    - COCO GT를 만들고 (pycocotools.Coco)
    - 모델 prediction을 전부 모은 뒤
    - eval_intersection_run(...)으로 IR / kernel metrics 계산.

    Returns:
        EvalIRResult(per_image, aggregates)
            aggregates 안에 ic@000/ic@030/ic@050/ic_multi/ic_ratio 등도 포함해서 반환.
    """
    Logger.info("[Eval] Building COCO API from dataset...")
    coco_gt = get_coco_api_from_dataset(data_loader.dataset)

    # prediction 리스트 수집
    preds = _collect_predictions_from_loader(
        model=model,
        dataloader=data_loader,
        device=device,
        score_thresh=0.0,
        max_dets=None,
        with_masks=True,
    )

    Logger.info("[Eval] Computing intersection / kernel metrics...")
    per_image, aggregates = eval_intersection_run(
        coco_gt=coco_gt,
        preds=preds,
        denom=denom,
        pred_agg=pred_agg,
        cat_id=cat_id,
        window_source=window_source,
        window_target=window_target,
        win_w=win_w,
        win_h=win_h,
    )

    # ic@000 / ic@030 / ic@050 / ic_multi / ic_ratio 추가
    ir_values = np.array([x.ir for x in per_image], dtype=float)
    aggregates = dict(aggregates)  # 복사해서 확장
    aggregates["ic@000"] = float(np.mean(ir_values > 0.0))
    aggregates["ic@030"] = float(np.mean(ir_values >= 0.30))
    aggregates["ic@050"] = float(np.mean(ir_values >= 0.50))
    aggregates["ic_multi"] = float(aggregates.get("multi_coverage", 0.0))
    aggregates["ic_ratio"] = float(aggregates.get("mean_ir", 0.0))

    Logger.info("[Eval] Done. mean_ir=%.4f, ic@050=%.4f",
                aggregates["mean_ir"], aggregates["ic@050"])

    return EvalIRResult(per_image=per_image, aggregates=aggregates)


# ----------------------------------------------------------------------
# 2) Standalone Intersection metrics (CLI)
# ----------------------------------------------------------------------


def _summarize_ic_row(
    name: str,
    denom: str,
    pred_agg: str,
    window_source: str,
    window_target: str,
    win_w: int,
    win_h: int,
    per_image: Sequence[ImageIR],
    agg: Dict[str, float],
) -> Dict[str, Any]:
    ir_values = np.array([x.ir for x in per_image], dtype=float)

    ic000 = float(np.mean(ir_values > 0.0))
    ic030 = float(np.mean(ir_values >= 0.30))
    ic050 = float(np.mean(ir_values >= 0.50))

    row: Dict[str, Any] = {
        "name": name,
        "denom": denom,
        "pred_agg": pred_agg,
        "kernel": f"{win_w}x{win_h}",
        "window_source": window_source,
        "window_target": window_target,
        "num_images": agg["num_images"],
        "ic@000": ic000,
        "ic@030": ic030,
        "ic@050": ic050,
        "ic_multi": float(agg["multi_coverage"]),
        "ic_ratio": float(agg["mean_ir"]),
        "mean_density": float(agg["mean_density"]),
        "mean_centeredness": float(agg["mean_centeredness"]),
        "mean_mixture": float(agg["mean_mixture"]),
        "median_ir": float(agg["median_ir"]),
        "p90_ir": float(agg["p90_ir"]),
    }
    return row


def _run_mode_predictions(args):
    """
    Mode A: GT JSON + prediction JSON(s) → intersection / kernel metrics.
    """
    os.makedirs(args.out, exist_ok=True)
    coco_gt = COCO(args.gt)

    pred_paths = args.pred
    names = args.name or []
    while len(names) < len(pred_paths):
        names.append(os.path.splitext(os.path.basename(pred_paths[len(names)]))[0])

    win_w, win_h = args.kernel

    all_rows: List[Dict[str, Any]] = []
    summary_json: Dict[str, Any] = {}

    for pred_path, name in zip(pred_paths, names):
        with open(pred_path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        if isinstance(preds, dict):
            raise ValueError(
                "Prediction file must be a list of COCO-style detections (not a dict)."
            )

        per_image, agg = eval_intersection_run(
            coco_gt=coco_gt,
            preds=preds,
            denom=args.denom,
            pred_agg=args.pred_agg,
            cat_id=args.cat_id,
            window_source=args.window_source,
            window_target=args.window_target,
            win_w=win_w,
            win_h=win_h,
        )

        row = _summarize_ic_row(
            name=name,
            denom=args.denom,
            pred_agg=args.pred_agg,
            window_source=args.window_source,
            window_target=args.window_target,
            win_w=win_w,
            win_h=win_h,
            per_image=per_image,
            agg=agg,
        )

        all_rows.append(row)
        summary_json[name] = row

        # per-image CSV (옵션)
        if args.per_image:
            import csv

            per_csv = os.path.join(args.out, f"{name}_per_image.csv")
            with open(per_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "image_id",
                        "width",
                        "height",
                        "ir",
                        "overlap_count",
                        "density",
                        "centeredness",
                        "mixture",
                    ],
                )
                w.writeheader()
                for it in per_image:
                    w.writerow(asdict(it))

        # per-batch kernel metrics (옵션)
        if args.batch_size and args.batch_size > 0:
            import csv

            dens_vals = np.array([x.density for x in per_image], dtype=float)
            cent_vals = np.array([x.centeredness for x in per_image], dtype=float)
            mix_vals = np.array([x.mixture for x in per_image], dtype=float)

            n = len(per_image)
            bs = int(args.batch_size)
            rows = []
            for i in range(0, n, bs):
                j = min(i + bs, n)
                rows.append(
                    {
                        "name": name,
                        "batch_index": i // bs,
                        "start_idx": i,
                        "end_idx": j - 1,
                        "batch_size": j - i,
                        "mean_density": float(np.mean(dens_vals[i:j])),
                        "mean_centeredness": float(np.mean(cent_vals[i:j])),
                        "mean_mixture": float(np.mean(mix_vals[i:j])),
                    }
                )
            batch_csv = os.path.join(args.out, f"{name}_batch_metrics.csv")
            with open(batch_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for r in rows:
                    w.writerow(r)

    # 요약 저장
    import csv

    csv_path = os.path.join(args.out, "summary.csv")
    if len(all_rows) == 0:
        raise RuntimeError("No predictions evaluated.")
    hdr = list(all_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    json_path = os.path.join(args.out, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print("\n=== Intersection & Kernel Metrics Summary ===")
    for row in all_rows:
        print(row)
    print(f"\nSaved summary CSV → {csv_path}")
    print(f"Saved summary JSON → {json_path}")


def _import_from_path(path: str):
    spec = importlib.util.spec_from_file_location("user_experiment", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _device_from_exp(exp_mod):
    if hasattr(exp_mod, "get_device"):
        return exp_mod.get_device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_mode_model(args):
    """
    Mode B: GT JSON + (experiment module, checkpoint)
    -> 내부에서 모델/데이터셋을 빌드해서 inference + 평가.
    Experiment module API 예시:

        def build_model(checkpoint: Optional[str] = None) -> torch.nn.Module
        def build_val_loader() -> torch.utils.data.DataLoader
        def get_device() -> torch.device    # optional
    """
    exp = _import_from_path(args.exp)

    device = _device_from_exp(exp)
    model = exp.build_model(checkpoint=args.checkpoint).to(device)
    val_loader = exp.build_val_loader()

    preds = _collect_predictions_from_loader(
        model=model,
        dataloader=val_loader,
        device=device,
        score_thresh=args.score_thresh,
        max_dets=args.max_dets,
        with_masks=not args.no_masks,
    )

    coco_gt = COCO(args.gt)
    win_w, win_h = args.kernel

    per_image, agg = eval_intersection_run(
        coco_gt=coco_gt,
        preds=preds,
        denom=args.denom,
        pred_agg=args.pred_agg,
        cat_id=args.cat_id,
        window_source=args.window_source,
        window_target=args.window_target,
        win_w=win_w,
        win_h=win_h,
    )

    row = _summarize_ic_row(
        name="model_eval",
        denom=args.denom,
        pred_agg=args.pred_agg,
        window_source=args.window_source,
        window_target=args.window_target,
        win_w=win_w,
        win_h=win_h,
        per_image=per_image,
        agg=agg,
    )

    os.makedirs(args.out, exist_ok=True)
    import csv

    csv_path = os.path.join(args.out, "summary.csv")
    hdr = list(row.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerow(row)

    json_path = os.path.join(args.out, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"model_eval": row}, f, ensure_ascii=False, indent=2)

    print("Evaluation result:", row)
    print(f"Saved → {csv_path} / {json_path}")


def _parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Observer Intersection/Kernel evaluator:\n"
            "  - import evaluate(model, loader, device) for training-time IR metrics\n"
            "  - use CLI for offline evaluation (predictions or model+dataset)"
        )
    )
    ap.add_argument("--gt", help="Path to COCO instances json (GT).")
    ap.add_argument("--out", default="./eval_ir_out", help="Output directory.")

    # Mode A (predictions)
    ap.add_argument(
        "--pred",
        action="append",
        help="Prediction json (COCO results). Can repeat.",
    )
    ap.add_argument(
        "--name",
        action="append",
        help="Name for each prediction (defaults to basename).",
    )

    # Mode B (model+dataset)
    ap.add_argument(
        "--exp",
        help=(
            "Path to experiment module .py implementing "
            "build_model()/build_val_loader()/[get_device()]."
        ),
    )
    ap.add_argument(
        "--checkpoint", default=None, help="Optional model checkpoint path."
    )
    ap.add_argument(
        "--score-thresh",
        type=float,
        default=0.0,
        help="Score threshold for detections.",
    )
    ap.add_argument(
        "--max-dets",
        type=int,
        default=None,
        help="Cap number of detections per image.",
    )
    ap.add_argument(
        "--no-masks",
        action="store_true",
        help="Do not export masks (bbox-only predictions).",
    )

    # 공통 metric 옵션
    ap.add_argument(
        "--denom",
        choices=["gt", "pred", "union"],
        default="gt",
        help="Denominator for IR.",
    )
    ap.add_argument(
        "--pred-agg",
        choices=["best", "union"],
        default="best",
        help="Aggregate predictions per image.",
    )
    ap.add_argument(
        "--cat-id", type=int, default=None, help="Filter category id (optional)."
    )
    ap.add_argument(
        "--per-image",
        action="store_true",
        help="(predictions mode) write per-image CSV as well.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="(predictions mode) per-batch means for kernel metrics.",
    )
    ap.add_argument(
        "--kernel",
        default="20,12",
        help="Window size W,H (pixels), e.g., '20,12'.",
    )
    ap.add_argument(
        "--window-source",
        choices=["pred", "gt"],
        default="pred",
        help="Center the kernel window on which mask.",
    )
    ap.add_argument(
        "--window-target",
        choices=["gt", "intersect"],
        default="gt",
        help="Target mask for kernel metrics.",
    )
    args = ap.parse_args()

    # kernel 파싱
    if args.kernel and isinstance(args.kernel, str):
        try:
            W, H = [int(x.strip()) for x in args.kernel.split(",")]
            args.kernel = (W, H)
        except Exception:
            raise ValueError(
                "--kernel must be 'W,H' with integers, e.g., 20,12"
            )

    return args


def main():
    """
    CLI entrypoint for Intersection metrics only.
    훈련 중 COCO mAP은 계산하지 않고, IR / kernel metric만 계산.
    """
    args = _parse_args()

    if args.pred:
        if args.gt is None:
            raise SystemExit("--gt is required when using --pred.")
        _run_mode_predictions(args)
    elif args.exp:
        if args.gt is None:
            raise SystemExit("--gt is required when using --exp.")
        _run_mode_model(args)
    else:
        raise SystemExit(
            "Please provide either --pred (prediction file[s]) "
            "or --exp (experiment module)."
        )


if __name__ == "__main__":
    main()
