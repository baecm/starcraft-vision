# src/pipeline.py
from __future__ import annotations

import os
import hydra
from omegaconf import DictConfig

from src.train import run_training
from src.inference import run_inference
from src.cli import parse_inference_args


def _build_inference_argv_from_cfg(cfg: DictConfig) -> list[str]:
    argv: list[str] = []

    # --replays
    test_replays = cfg.get("test_replays")
    if not test_replays:
        raise ValueError("cfg.test_replays 가 비어 있어서 inference를 할 replays를 알 수 없습니다.")
    argv += ["--replays", *[str(r) for r in test_replays]]

    # 경로 / 모델 정보
    argv += [
        "--data-root", cfg.data_root,
        "--output-dir", os.path.join(cfg.log_root, "predictions"),
        "--model-root", cfg.log_root,
        "--model-name", cfg.id_string,
        "--model-number", str(cfg.max_epoch),
        "--label-method", cfg.label_method,
        "--window-size", str(cfg.window_size),
        "--sample-ratio", str(cfg.sample_ratio),
    ]

    # run-name (없으면 기본값 사용)
    run_name = getattr(cfg, "inference_run_name", None)
    if run_name is None:
        run_name = os.path.join(cfg.id_string, f"model_{cfg.max_epoch:03d}")
    argv += ["--run-name", run_name]

    # include-components
    if cfg.get("include_components"):
        argv += ["--include-components", *cfg.include_components]

    # 하이퍼파라미터
    batch_size = getattr(cfg, "inference_batch_size", cfg.batch_size)
    workers = cfg.num_workers
    score_threshold = getattr(cfg, "score_threshold", 0.5)

    argv += [
        "--batch-size", str(batch_size),
        "--workers", str(workers),
        "--score-threshold", str(score_threshold),
    ]

    # 플래그
    if cfg.cuda:
        argv.append("--cuda")
    if getattr(cfg, "use_kbrs", False):
        argv.append("--use-kbrs")

    if cfg.get("seed") is not None:
        argv += ["--seed", str(cfg.seed)]

    return argv


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("[2025-12-01T..] Runninng train/inference squentially...")

    mode = getattr(cfg, "mode", "train_only")

    # 1) train
    if mode in ("train_only", "train_and_inference"):
        run_training(cfg)

    # 2) inference
    if mode in ("inference_only", "train_and_inference"):
        inf_argv = _build_inference_argv_from_cfg(cfg)
        inf_args = parse_inference_args(inf_argv)
        run_inference(inf_args)


if __name__ == "__main__":
    main()
