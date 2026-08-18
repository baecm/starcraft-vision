# src/profile_kbrs_forward.py
# 목적:
# - "input 1장 기준" KBRS scorer 호출 시간(= model.kbrs_scorer(fmap_for_kbrs))을 측정
# - batch size를 1/8/16/32로 바꿔가며 ms/call, ms/img 통계 출력
# - 실험 환경(conf)과 최대한 동일하게 builder(get_model_instance_segmentation) 사용
#
# 실행 예:
#   PYTHONPATH=./src python src/profile_kbrs_scorer.py
#   (컨테이너 내부면 경로에 맞게)

import time
from collections import OrderedDict

import numpy as np
import torch

from model.maskrcnn_builder import get_model_instance_segmentation
from model.plugins.kbrs import KBRSConvScorer
from model.utils import pick_feature_map, normalize_projections, auto_expand_indices, compute_gate_from_raw_inputs
from utils.time_measure import measure_time


def bench_call_ms(fn, iters=200, warmup=30, sync_cuda=True):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    times_ms = []
    with torch.no_grad():
        for _ in range(iters):
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if iters > 1 else 0.0,
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "iters": iters,
        "warmup": warmup,
    }

def print_row(prefix, bs, s):
    mean = s["mean"]
    p50 = s.get("p50", None)
    p95 = s.get("p95", None)
    p99 = s.get("p99", None)

    pct_str = ""
    if p50 is not None and p95 is not None and p99 is not None:
        pct_str = f"  p50={p50:8.3f}  p95={p95:8.3f}  p99={p99:8.3f}"

    print(
        f"{prefix}  bs={bs:<2d}  "
        f"mean={mean:8.3f} ms/call  mean={mean/bs:8.3f} ms/img"
        f"{pct_str}",
        flush=True,
    )

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device, flush=True)

    # ===== conf 기반 실험 조건 =====
    window_size = 4
    per_window = 9
    C = window_size * per_window  # 36
    H, W = 128, 128

    # KBRS params (conf + 네 기존 설정 반영)
    kbrs_params = {
        "feature_map_name": "smallest",
        "scorer_impl": "conv",
        "detach_scorer_input": False,
        "region_size": (20, 12),
        "score_stride": 1,
        "mixture_tau": 2.0,
        "mixture_mode": "confusion",
        "mixture_power": 1.0,
        "tau": 4.0,
        "use_entropy": False,
        # score weights (중요: KBRS_MaskRCNN.py는 score_weights가 아니라 이 키들을 읽음)
        "density": 0.3,
        "mixture": 3.0,
        "centeredness": 0.3,
        # projections / gate (fallback mixture 크래시 방지 + 의미 유지)
        "projections": [{"name": "A", "channels": [0, 1, 2, 3]},
                        {"name": "B", "channels": [4, 5, 6, 7]}],
        "mixture_between": ("A", "B"),
        "gate_channels": [8],
        "gate_reduce": "mean",
        "gate_gain": 0.8,
        # logging/viz는 profiling에선 off 권장
        "viz_components": False,
        "accumulate_epoch": False,
        "component_losses": False,
        # builder 주입 메타도 명시(안 해도 되지만 명확성)
        "window_size": window_size,
        "per_window": per_window,
    }

    loss_weights = {"loss_kbrs": 0.25}

    # ===== 모델 생성 (실험 환경에 맞추기 위해 builder 사용) =====
    model = get_model_instance_segmentation(
        num_classes=2,
        window_size=window_size,
        in_channels=C,
        do_normalize=False,     # conf
        resize_mode="resize",   # conf
        min_sizes=[800],        # conf
        max_size=1333,          # conf
        use_kbrs=True,
        kbrs_params=kbrs_params,
        loss_weights=loss_weights,
    ).to(device).eval()

    print(f"Using {C} input channels (window size: {window_size}, per_window: {per_window})", flush=True)
    print("MODEL CLASS =", model.__class__.__name__, flush=True)

    # ===== dummy raw input (batch=1) =====
    raw_images_1 = [torch.randn(C, H, W, device=device)]

    # transform/backbone/fmap 준비 (1회)
    with torch.no_grad():
        with measure_time("transform (1x)"):
            images_t, _ = model.transform(raw_images_1, targets=None)

        with measure_time("backbone (1x)"):
            features = model.backbone(images_t.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([("0", features)])

        with measure_time("pick feature map (1x)"):
            _, fmap = pick_feature_map(features, model.kbrs_params.get("feature_map_name", "smallest"))

    print("fmap shape =", tuple(fmap.shape), flush=True)

    # ===== scorer init (1x) + call 준비 =====
    in_channels_total = model._per_window * model._window_size
    proj_norm = normalize_projections(
        model._projections_cfg, model._window_size, model._per_window, in_channels_total
    )

    if model.kbrs_scorer is None:
        with measure_time("KBRSConvScorer init (1x)"):
            model.kbrs_scorer = KBRSConvScorer(
                region_size=model._scorer_region_size,
                weights=model._weights,
                projections=proj_norm,
                mixture_tau=model._mixture_tau,
                mixture_mode=model._mixture_mode,
                mixture_power=model._mixture_power,
                mask_channel=None,
                mask_gain=1.0,
                score_stride=model._scorer_stride,
                downsample_before=model._downsample_before,
            ).to(fmap.device, dtype=fmap.dtype)

    fmap_for_kbrs_1 = fmap.detach() if model._detach_scorer_input else fmap

    # ===== batch sweep: scorer call only =====
    print("\n== KBRS scorer CALL only: kbrs_scorer(fmap_for_kbrs) ==", flush=True)
    batch_sizes = [1, 8, 16, 32]
    
    CALL_ITERS = 10000
    CALL_WARMUP = 200
    PART_ITERS = 1000
    PART_WARMUP = 100

    for bs in batch_sizes:
        try:
            fmap_bs = fmap_for_kbrs_1.repeat(bs, 1, 1, 1).contiguous()
            scorer = model.kbrs_scorer

            # 0) 전체 scorer forward
            s = bench_call_ms(lambda: scorer(fmap_bs), iters=CALL_ITERS, warmup=CALL_WARMUP)
            print_row("CALL total", bs, s)

            # 2.A downsample (옵션)
            sA = bench_call_ms(lambda: scorer._maybe_downsample(fmap_bs), iters=PART_ITERS, warmup=PART_WARMUP)
            print_row("2.A downsample", bs, sA)

            # downsample 결과를 실제로 쓰도록(공정)
            x_ds = scorer._maybe_downsample(fmap_bs)

            # 2.B density
            sB = bench_call_ms(lambda: scorer._density(x_ds), iters=PART_ITERS, warmup=PART_WARMUP)
            print_row("2.B density", bs, sB)

            # 2.C centeredness
            sC = bench_call_ms(lambda: scorer._centeredness(x_ds), iters=PART_ITERS, warmup=PART_WARMUP)
            print_row("2.C centeredness", bs, sC)

            # 2.D mixture
            sD = bench_call_ms(lambda: scorer._mixture_from_projections(x_ds), iters=PART_ITERS, warmup=PART_WARMUP)
            print_row("2.D mixture", bs, sD)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"CALL  bs={bs}  OOM (skip)", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
    print("\nDone.", flush=True)
        



if __name__ == "__main__":
    main()
