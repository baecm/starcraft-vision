# src/evaluate.py
from __future__ import annotations

import os
import csv
import json
import argparse

from dataclasses import asdict
from typing import Any, Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from custom_evaluator import eval_intersection_run, ImageIR
from utils.logger import Logger


# ----------------------------------------------------------------------
# Helper: compute centroid from COCO annotation or detection
# ----------------------------------------------------------------------
def _centroid_from_coco_ann(ann: dict, img_w: int, img_h: int) -> Tuple[float, float]:
    """
    Return centroid (cx, cy) in image pixel coordinates (0..img_w-1, 0..img_h-1)
    Handles 'segmentation' (polygon or RLE) and 'bbox'.
    """
    if "segmentation" in ann and ann["segmentation"]:
        seg = ann["segmentation"]
        # get bbox from segmentation then centroid from bbox center
        if isinstance(seg, dict) and "counts" in seg:
            bbox = mask_util.toBbox(seg)  # (x,y,w,h) float
        else:
            try:
                rles = mask_util.frPyObjects(seg, img_h, img_w) if isinstance(seg, list) else seg
                mrg = mask_util.merge(rles)
                bbox = mask_util.toBbox(mrg)
            except Exception:
                bbox = None
        if bbox is not None:
            x, y, w, h = bbox
            return float(x + w / 2.0), float(y + h / 2.0)

    if "bbox" in ann and ann["bbox"]:
        x, y, w, h = ann["bbox"]
        return float(x + w / 2.0), float(y + h / 2.0)

    # fallback: if segmentation/bbox absent, try keypoints? else center of image
    return float(img_w) / 2.0, float(img_h) / 2.0


# ----------------------------------------------------------------------
# Convert COCO GT + COCO predictions into "agent traces" format
# Each image -> one test; agents: [predictor (one), GT objects...],
# each agent is a list with single frame dict {"vpx","vpy"}.
# ----------------------------------------------------------------------
def coco_to_kernel_labels(
    coco_gt: COCO,
    preds_list: List[dict],
    *,
    x_len: int,
    y_len: int,
    grid_w: int,
    grid_h: int,
    max_x: float,
    max_y: float,
) -> List[List[List[Dict[str, float]]]]:
    """
    Convert COCO-style GT + preds into agent-trace tests for kernel-based evaluator.

    Args:
        coco_gt: COCO object loaded from GT json
        preds_list: list of detection dicts (COCO results), each with 'image_id',
                    'bbox' or 'segmentation', optional 'score'
        x_len,y_len,grid_w,grid_h,max_x,max_y: mapping parameters

    Returns:
        tests: list of tests; each test is list of agents;
               each agent is a list of frames(dict with vpx/vpy)
    """
    # group preds by image_id
    preds_by_img: Dict[int, List[dict]] = {}
    for p in preds_list:
        img_id = int(p["image_id"])
        preds_by_img.setdefault(img_id, []).append(p)

    tests: List[List[List[Dict[str, float]]]] = []

    for img in coco_gt.dataset.get("images", []):
        image_id = int(img["id"])
        img_w = int(img.get("width", grid_w))
        img_h = int(img.get("height", grid_h))

        # get GT annotations for this image
        ann_ids = coco_gt.getAnnIds(imgIds=image_id)
        anns = coco_gt.loadAnns(ann_ids) if ann_ids else []

        # if no predictions for this image, skip (we cannot evaluate predictor)
        img_preds = preds_by_img.get(image_id, [])
        if len(img_preds) == 0:
            continue

        # Choose predictor: highest score if available, else first pred
        if any("score" in p for p in img_preds):
            best_pred = max(img_preds, key=lambda q: float(q.get("score", 0.0)))
        else:
            best_pred = img_preds[0]

        # compute centroid pixel coords for predictor
        pcx, pcy = _centroid_from_coco_ann(best_pred, img_w, img_h)

        # convert pixel centroid to 'vpx','vpy' (map-scale coords) so that
        # the kernel-mapping recovers px,py consistently:
        # original mapping: px = round(vx/max_x * (width - x_len))
        # invert: vx = px / (width - x_len) * max_x
        vx = float(pcx) / max(1, (img_w - x_len)) * max_x
        vy = float(pcy) / max(1, (img_h - y_len)) * max_y

        # agent0: predictor with single frame
        agent0 = [{"vpx": vx, "vpy": vy}]

        # agents 1..: one per GT annotation (centroid mapped similarly)
        ref_agents: List[List[Dict[str, float]]] = []
        for ann in anns:
            gcx, gcy = _centroid_from_coco_ann(ann, img_w, img_h)
            gvx = float(gcx) / max(1, (img_w - x_len)) * max_x
            gvy = float(gcy) / max(1, (img_h - y_len)) * max_y
            ref_agents.append([{"vpx": gvx, "vpy": gvy}])

        # Compose test: predictor first, then references
        test_agents: List[List[Dict[str, float]]] = [agent0] + ref_agents

        # Append test only if at least one reference exists
        if len(ref_agents) == 0:
            continue

        tests.append(test_agents)

    return tests


# ----------------------------------------------------------------------
# Public helper: evaluate kernel metrics directly from COCO objects
# (for using inside train/eval code, without going through JSON files)
# ----------------------------------------------------------------------
def eval_kernel_from_coco(
    coco_gt: COCO,
    preds_list: List[dict],
    *,
    name: str = "run",
    kernel: Tuple[int, int] = (20, 12),
    grid: Tuple[int, int] = (128, 128),
    maxcoord: Tuple[float, float] = (3456.0, 3720.0),
) -> Tuple[Dict[str, Any], List[ImageIR], Dict[str, float]]:
    """
    Convenience wrapper for train/eval code.

    Returns:
        summary_row: dict with ic@000/ic@030/ic@050/ic_multi/ic_ratio/... for this run
        per_image: list[ImageIR]
        agg: aggregated stats from eval_intersection_run
    """
    x_len, y_len = kernel
    width, height = grid
    max_x, max_y = maxcoord

    labels_tests = coco_to_kernel_labels(
        coco_gt=coco_gt,
        preds_list=preds_list,
        x_len=x_len,
        y_len=y_len,
        grid_w=width,
        grid_h=height,
        max_x=max_x,
        max_y=max_y,
    )

    per_image, agg = eval_intersection_run(
        labels_tests,
        x_len=x_len,
        y_len=y_len,
        width=width,
        height=height,
        max_x=max_x,
        max_y=max_y,
    )

    row = _summarize_ic_row(
        name=name,
        kernel=(x_len, y_len),
        per_image=per_image,
        agg=agg,
    )

    try:
        from utils.multi_region_eval import compute_multi_region_metrics
        multi_metrics = compute_multi_region_metrics(
            coco_gt=coco_gt,
            preds_list=preds_list,
            grid_w=width,
            grid_h=height,
            win_w=x_len,
            win_h=y_len,
        )
        row.update(multi_metrics)
    except Exception as e:
        Logger.warn(f"[MultiRegion] Metric calculation skipped due to: {e}")

    return row, per_image, agg


def _summarize_ic_row(
    name: str,
    kernel: Tuple[int, int],
    per_image: Sequence[ImageIR],
    agg: Dict[str, float],
) -> Dict[str, Any]:
    ir_values = np.array([x.ir for x in per_image], dtype=float)
    ic000 = float(np.mean(ir_values > 0.0)) if ir_values.size > 0 else 0.0
    ic030 = float(np.mean(ir_values >= 0.30)) if ir_values.size > 0 else 0.0
    ic050 = float(np.mean(ir_values >= 0.50)) if ir_values.size > 0 else 0.0

    row: Dict[str, Any] = {
        "name": name,
        "kernel": f"{kernel[0]}x{kernel[1]}",
        "num_images": agg.get("num_images", 0),
        "ic@000": ic000,
        "ic@030": ic030,
        "ic@050": ic050,
        "ic_multi": float(agg.get("multi_intersection", 0.0)),
        "ic_ratio": float(agg.get("mean_ir", 0.0)),
        "median_ir": float(agg.get("median_ir", 0.0)),
        "p90_ir": float(agg.get("p90_ir", 0.0)),
    }
    return row


# ----------------------------------------------------------------------
# Core runner (reusable from train / other scripts)
# ----------------------------------------------------------------------
def run_kernel_eval(
    *,
    gt_path: Optional[str],
    gt_dir: str,
    pred_files: Sequence[str],
    pred_dir: str,
    out_dir: str,
    names: Sequence[str],
    kernel: Tuple[int, int],
    grid: Tuple[int, int],
    maxcoord: Tuple[float, float],
    per_image_csv: bool = False,
    batch_size: int = 0,
    run_tag: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], str, str]:
    """
    Run kernel-based intersection evaluation for one or more prediction files.

    Returns:
        all_rows: list of summary rows (one per prediction name)
        summary_json: mapping name -> summary row
        csv_path: path to summary CSV
        json_path: path to summary JSON
    """
    os.makedirs(out_dir, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    summary_json: Dict[str, Dict[str, Any]] = {}

    # If GT provided, load once
    coco_gt = None
    if gt_path is not None:
        full_gt_path = gt_path if os.path.isabs(gt_path) else os.path.join(gt_dir, gt_path)
        if not os.path.isfile(full_gt_path):
            raise FileNotFoundError(f"--gt file not found: {full_gt_path}")
        coco_gt = COCO(full_gt_path)
        Logger.info(f"Loaded GT COCO: {full_gt_path} (images={len(coco_gt.dataset.get('images', []))})")

    x_len, y_len = kernel
    width, height = grid
    max_x, max_y = maxcoord

    for pred_path, name in zip(pred_files, names):
        path = os.path.join(pred_dir, pred_path)
        if os.path.isabs(pred_path):
            path = pred_path
        else:
            path = os.path.join(pred_dir, pred_path)

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Prediction file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        # If loaded already in tests format, use directly
        if isinstance(loaded, dict) and "tests" in loaded:
            labels_tests = loaded["tests"]
        elif isinstance(loaded, list) and len(loaded) > 0 and isinstance(loaded[0], list):
            labels_tests = loaded
        elif coco_gt is not None:
            # loaded expected to be COCO detection results
            preds_list = loaded
            if isinstance(loaded, dict) and "annotations" in loaded:
                preds_list = loaded["annotations"]
            if not isinstance(preds_list, list):
                raise ValueError("Prediction file not understood: expected tests or COCO detection list.")
            labels_tests = coco_to_kernel_labels(
                coco_gt=coco_gt,
                preds_list=preds_list,
                x_len=x_len,
                y_len=y_len,
                grid_w=width,
                grid_h=height,
                max_x=max_x,
                max_y=max_y,
            )
        else:
            raise ValueError(
                "Prediction file not in tests format and --gt not provided to convert COCO results."
            )

        # Run evaluator
        per_image, agg = eval_intersection_run(
            labels_tests,
            x_len=x_len,
            y_len=y_len,
            width=width,
            height=height,
            max_x=max_x,
            max_y=max_y,
        )

        # summarize row
        row = _summarize_ic_row(name=name, kernel=(x_len, y_len), per_image=per_image, agg=agg)
        all_rows.append(row)
        summary_json[name] = row

        # per-image CSV
        if per_image_csv:

            per_csv = os.path.join(out_dir, f"{name}_per_image.csv")
            with open(per_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "image_id",
                        "width",
                        "height",
                        "ir",
                        "overlap_count",
                    ],
                )
                w.writeheader()
                for it in per_image:
                    w.writerow(asdict(it))

        # per-batch kernel metrics (optional)
        if batch_size and batch_size > 0 and len(per_image) > 0:

            n = len(per_image)
            bs = int(batch_size)
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
                    }
                )
            batch_csv = os.path.join(out_dir, f"{name}_batch_metrics.csv")
            with open(batch_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for r in rows:
                    w.writerow(r)

    if len(all_rows) == 0:
        raise RuntimeError("No predictions evaluated.")

    base_name = "summary"
    if run_tag:
        base_name = f"{base_name}_{run_tag}"

    csv_path = os.path.join(out_dir, f"{base_name}.csv")
    hdr = list(all_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    json_path = os.path.join(out_dir, f"{base_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    return all_rows, summary_json, csv_path, json_path


class KernelEvalResult:
    """
    train.py에서 쓰기 위한 간단한 래퍼:
    - .aggregates 딕셔너리만 있으면 train 루프가 그대로 돌 수 있음
    """
    def __init__(self, aggregates: Dict[str, Any]):
        self.aggregates = aggregates


def evaluate(model, data_loader, device, epoch: int = 0):
    """
    Train 중에 호출되는 evaluate 함수.

    - COCO mAP은 계산하지 않고
    - model + data_loader에서 바로 prediction/GT를 읽어서
    - Observer kernel 기반 IC metric만 계산한다.
    - 반환값은 .aggregates 딕셔너리를 가진 KernelEvalResult 객체.
    """
    model.eval()

    # ---- kernel/grid/maxcoord 설정 ----
    # KBRS_PARAMS.region_size를 우선 사용하고, 없으면 디폴트 (20,12)
    try:
        import config as _cfg
        if hasattr(_cfg, "KBRS_PARAMS") and "region_size" in _cfg.KBRS_PARAMS:
            kernel = tuple(_cfg.KBRS_PARAMS["region_size"])
        else:
            kernel = (20, 12)
    except Exception:
        kernel = (20, 12)

    # grid / maxcoord는 기존 스크립트 기본값과 동일하게
    grid = (128, 128)
    maxcoord = (3456.0, 3720.0)

    x_len, y_len = kernel
    width, height = grid
    max_x, max_y = maxcoord

    labels_tests: List[List[List[Dict[str, float]]]] = []

    with torch.no_grad():
        for images, targets in data_loader:
            # images: List[Tensor[C,H,W]]
            # targets: List[Dict]
            images = [img.to(device) for img in images]
            outputs = model(images)

            for img, tgt, out in zip(images, targets, outputs):
                # img 크기 (모델 입력 기준, bbox도 여기에 맞춰져 있음)
                _, img_h, img_w = img.shape

                boxes_pred = out["boxes"]
                scores_pred = out["scores"]

                # 예측이 없으면 스킵
                if boxes_pred.numel() == 0:
                    continue

                # 최고 score 1개만 사용 (이전 설계와 동일하게 "대표 뷰포트"로 봄)
                best_idx = int(scores_pred.argmax().item())
                px1, py1, px2, py2 = boxes_pred[best_idx].detach().cpu().tolist()
                pw = px2 - px1
                ph = py2 - py1
                pcx = px1 + pw / 2.0
                pcy = py1 + ph / 2.0

                # pixel → vpx, vpy (기존 coco_to_kernel_labels 와 동일한 역변환)
                vx = float(pcx) / max(1, (img_w - x_len)) * max_x
                vy = float(pcy) / max(1, (img_h - y_len)) * max_y
                agent0 = [{"vpx": vx, "vpy": vy}]

                # GT 박스들 (하나 이상 있을 수 있음)
                gt_boxes = tgt["boxes"].detach().cpu().numpy()
                ref_agents: List[List[Dict[str, float]]] = []

                for gx1, gy1, gx2, gy2 in gt_boxes:
                    gw = gx2 - gx1
                    gh = gy2 - gy1
                    gcx = gx1 + gw / 2.0
                    gcy = gy1 + gh / 2.0
                    gvx = float(gcx) / max(1, (img_w - x_len)) * max_x
                    gvy = float(gcy) / max(1, (img_h - y_len)) * max_y
                    ref_agents.append([{"vpx": gvx, "vpy": gvy}])

                if len(ref_agents) == 0:
                    # GT 없으면 intersection 계산 불가 → 스킵
                    continue

                labels_tests.append([agent0] + ref_agents)

    # labels_tests 가 하나도 없으면 그냥 빈 결과 반환
    if len(labels_tests) == 0:
        Logger.info("[IC] No valid labels_tests (no preds or no GT). Skipping IC evaluation.")
        return KernelEvalResult(aggregates={})

    # ---- kernel evaluator 호출 ----
    per_image, agg = eval_intersection_run(
        labels_tests,
        x_len=x_len,
        y_len=y_len,
        width=width,
        height=height,
        max_x=max_x,
        max_y=max_y,
    )

    # 요약 row (ic@000, ic_ratio 등)
    row = _summarize_ic_row(
        name=f"val_epoch_{epoch:03d}",
        kernel=(x_len, y_len),
        per_image=per_image,
        agg=agg,
    )

    # train.py에서는 eval_stats.aggregates 를 보고 로그를 남기므로,
    # agg + row를 합쳐서 aggregates로 넣어준다.
    aggregates: Dict[str, Any] = {}
    aggregates.update(agg)
    aggregates.update(row)

    Logger.info(f"[IC] epoch={epoch} metrics={row}")

    return KernelEvalResult(aggregates=aggregates)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Kernel-based intersection evaluator driven from COCO GT + COCO predictions.\n"
            "If --gt is provided (COCO instances json), preds (COCO detection results)\n"
            "are converted and evaluated.\n"
            "Alternatively, --pred may contain JSON with 'tests' already and will be used directly."
        )
    )
    parser.add_argument(
        "--gt",
        default=None,
        help="Optional GT COCO json (instances). If provided, used to build tests.",
    )
    parser.add_argument(
        "--gt-dir",
        default=os.path.join(os.getcwd(), "data", "label", "dst"),
        help="Directory containing GT file.",
    )
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help="Prediction JSON file(s). Can be COCO results or tests JSON. Repeatable.",
    )
    parser.add_argument(
        "--pred-dir",
        default=os.path.join(os.getcwd(), "predictions"),
        help="Directory containing prediction files.",
    )
    parser.add_argument(
        "--out",
        default="./results",
        help="Output directory.",
    )
    parser.add_argument(
        "--name",
        action="append",
        help="Name for each prediction (defaults to basename).",
    )
    parser.add_argument(
        "--kernel",
        default="20,12",
        help="Window size x_len,y_len. Example: '20,12'.",
    )
    parser.add_argument(
        "--grid",
        default="128,128",
        help="Grid width,height representing sampling grid. Example: '128,128'.",
    )
    parser.add_argument(
        "--maxcoord",
        default="3456,3720",
        help="Original coordinate maxima (max_x,max_y) used for normalization.",
    )
    parser.add_argument(
        "--per-image",
        action="store_true",
        help="Write per-image CSV of results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="(optional) batch size for computing per-batch kernel means.",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help=(
            "Suffix for summary filenames (e.g. 'maskrcnn_kbrs_e050_replay-KR-7702711227'). "
            "Output will be summary_<run-tag>.csv/json."
        ),
    )

    args = parser.parse_args(argv)
    return args

def main():
    args = parse_args()
    
    all_rows, summary_json, csv_path, json_path = run_kernel_eval(
        gt_path=args.gt,
        gt_dir=args.gt_dir,
        pred_files=args.pred,
        pred_dir=args.pred_dir,
        out_dir=args.out,
        names=args.name,
        kernel=args.kernel,
        grid=args.grid,
        maxcoord=args.maxcoord,
        per_image_csv=args.per_image,
        batch_size=args.batch_size,
        run_tag=args.run_tag,
    )

    print("\n=== Intersection & Kernel Metrics Summary ===")
    for row in all_rows:
        print(row)
    print(f"\nSaved summary CSV : {csv_path}")
    print(f"\nSaved summary JSON : {json_path}")


if __name__ == "__main__":
    main()
