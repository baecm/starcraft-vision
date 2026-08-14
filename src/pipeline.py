from __future__ import annotations

import os
import hydra
from omegaconf import DictConfig

from train import run_training
from inference import run_inference
from cli import parse_inference_args


def _build_inference_argv_from_cfg(cfg: DictConfig) -> list[str]:
    argv: list[str] = []

    # 1. Dataset & Paths (데이터 및 경로)
    test_replays = cfg.get("test_replays")
    if not test_replays:
        raise ValueError("[Error] cfg.test_replays가 비어 있습니다. Inference를 수행할 리플레이가 필요합니다.")
    
    argv.extend(["--replays"] + [str(r) for r in test_replays])
    argv.extend([
        "--data-root", str(cfg.data_root),
        "--output-dir", str(cfg.prediction_root),
    ])

    # 2. Model & Checkpoint Info (모델 및 체크포인트 정보)
    argv.extend([
        "--model-root", str(cfg.model_root),
        "--model-name", str(cfg.id_string),
        "--model-number", str(cfg.max_epoch),
    ])

    # RT-DETR 전용 설정
    arch_cfg = cfg.get("architecture", {})
    argv.extend([
        "--rtdetr-version", str(arch_cfg.get("rtdetr_version", "v1")),
        "--rtdetr-size", str(arch_cfg.get("rtdetr_size", "l")),
    ])

    # 3. KBRS & Preprocessing Settings (전처리 및 공통 설정)
    argv.extend([
        "--label-method", str(cfg.label_method),
        "--window-size", str(cfg.window_size),
        "--sample-ratio", str(cfg.sample_ratio),
    ])

    if cfg.get("include_components"):
        argv.extend(["--include-components"] + list(cfg.include_components))

    # 4. Hyperparameters (추론 하이퍼파라미터)
    argv.extend([
        "--batch-size", str(cfg.get("inference_batch_size", cfg.batch_size)),
        "--workers", str(cfg.num_workers),
        "--score-threshold", str(cfg.get("score_threshold", 0.5)),
    ])

    # 5. Run Name (저장될 폴더명)
    # inference_run_name이 없으면 기본값으로 "id_string/model_00X" 형태 생성
    run_name = cfg.get("inference_run_name") or os.path.join(cfg.id_string, f"model_{cfg.max_epoch:03d}")
    argv.extend(["--run-name", str(run_name)])

    # 6. Flags (CUDA, KBRS, Seed)
    if cfg.get("cuda", True):
        argv.append("--cuda")
        
    use_kbrs_flag = cfg.get("kbrs", {}).get("use_kbrs", cfg.get("use_kbrs", False))
    if use_kbrs_flag:
        argv.append("--use-kbrs")
        
    if cfg.get("seed") is not None:
        argv.extend(["--seed", str(cfg.seed)])

    return argv


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("[Pipeline] Running train/inference sequentially...")

    mode = cfg.get("mode", "train_only")

    # 1) Train
    if mode in ("train_only", "train_and_inference"):
        run_training(cfg)

    # 2) Inference
    if mode in ("inference_only", "train_and_inference"):
        inf_argv = _build_inference_argv_from_cfg(cfg)
        inf_args = parse_inference_args(inf_argv)
        run_inference(inf_args)


if __name__ == "__main__":
    main()