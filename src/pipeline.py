"""
Train, then run inference on the test replays with the checkpoint just written (make run).

The inference half is driven through the same argument parser as
src/inference.py, with its options derived from the Hydra config here.
Note: unlike train.py, this entry point does not apply prepare_cfg, so
roci=true is ignored under make run; use make train for ROCI runs.
"""
from __future__ import annotations

import os
import hydra
from omegaconf import DictConfig

from train import run_training
from inference import run_inference
from cli import parse_inference_args


def _build_inference_argv_from_cfg(cfg: DictConfig) -> list[str]:
    argv: list[str] = []

    # 1. Replays and paths
    test_replays = cfg.get("test_replays")
    if not test_replays:
        raise ValueError("[Error] cfg.test_replays is empty; inference needs replays to run on.")
    
    argv.extend(["--replays"] + [str(r) for r in test_replays])
    argv.extend([
        "--data-root", str(cfg.data_root),
        "--output-dir", str(cfg.prediction_root),
    ])

    # 2. Which checkpoint
    argv.extend([
        "--model-root", str(cfg.model_root),
        "--model-name", str(cfg.id_string),
        "--model-number", str(cfg.max_epoch),
    ])

    # Architecture and the knobs that change what inference builds or emits.
    # These used to be omitted, so inference fell back to guessing the
    # architecture from id_string and to config.py defaults for the rest.
    arch_cfg = cfg.get("architecture", {})
    if arch_cfg.get("model_name"):
        argv.extend(["--architecture", str(arch_cfg.get("model_name"))])
    if arch_cfg.get("k_max") is not None:
        argv.extend(["--k-max", str(arch_cfg.get("k_max"))])
    if arch_cfg.get("conf_threshold") is not None:
        argv.extend(["--conf-threshold", str(arch_cfg.get("conf_threshold"))])

    # 3. Input settings
    argv.extend([
        "--label-method", str(cfg.label_method),
        "--window-size", str(cfg.window_size),
        "--sample-ratio", str(cfg.sample_ratio),
    ])

    if cfg.get("include_components"):
        argv.extend(["--include-components"] + list(cfg.include_components))

    # 4. Batching and the score filter
    argv.extend([
        "--batch-size", str(cfg.get("inference_batch_size", cfg.batch_size)),
        "--workers", str(cfg.num_workers),
        "--score-threshold", str(cfg.get("score_threshold", 0.5)),
    ])

    # 5. Output folder under --output-dir
    # defaults to <id_string>/model_<max_epoch>
    run_name = cfg.get("inference_run_name") or os.path.join(cfg.id_string, f"model_{cfg.max_epoch:03d}")
    argv.extend(["--run-name", str(run_name)])

    # 6. Flags
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
    import sys
    from config import resolve_cli_aliases
    sys.argv[1:] = resolve_cli_aliases(sys.argv[1:])
    main()