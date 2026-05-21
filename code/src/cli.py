# src/cli.py
import argparse
import os
import code.src.config as config
import yaml

from collections.abc import Iterable

DEFAULT_COMPONENTS = ["worker", "ground", "air", "building", "vision"]

def _flatten_list(x):
    """train_replays: [*set1, *set2] 처럼 list 안에 list 가 있을 때 평탄화."""
    if isinstance(x, (str, bytes)):
        return [x]
    if not isinstance(x, Iterable):
        return [x]
    out = []
    for v in x:
        if isinstance(v, (list, tuple)):
            out.extend(_flatten_list(v))
        else:
            out.append(v)
    return out

def _flatten_kbrs_params(prefix, node):
    for k, v in node.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from _flatten_kbrs_params(key, v)
        else:
            # 최종적으로 "score_weights.density=0.3" 이런 string
            yield f"{key}={v}"

# -------------------------
# 공통 group builder
# -------------------------
def add_data_args(
    parser: argparse.ArgumentParser,
    *,
    with_replays: bool = True,
    with_label: bool = False,
):
    """
    공통 데이터 관련 인자 그룹 (inference 등에서 사용).

    train 전용(--train-replay/--test-replay)은 parse_train_args 안에서 따로 정의하고,
    여기서는 단순 "--replays" 케이스(예: inference)를 위해 사용.
    """
    group = parser.add_argument_group("Data and Labeling")

    if with_replays:
        group.add_argument(
            "--replays",
            type=str,
            nargs="+",
            required=True,
            help="List of replay IDs.",
        )

    group.add_argument(
        "--data-root",
        type=str,
        default=os.path.join(os.getcwd(), "data"),
        help="Root directory for data.",
    )
    group.add_argument(
        "--include-components",
        type=str,
        nargs="+",
        default=DEFAULT_COMPONENTS,
        help="List of components to include.",
    )

    if with_label:
        group.add_argument(
            "--label-method",
            type=str,
            default=config.LABEL_METHODS[0],
            choices=config.LABEL_METHODS,
            help="Label extraction method (folder name).",
        )
        group.add_argument(
            "--sample-ratio",
            type=float,
            default=1.0,
            help="Fraction of dataset to sample.",
        )

    return group


def add_transform_args(parser: argparse.ArgumentParser):
    group_tf = parser.add_argument_group("Transform / Resize / Normalize")
    group_tf.add_argument(
        "--resize-mode",
        type=str,
        choices=["resize", "keep"],
        default="resize",
        help="'resize'면 old 스타일(권장), 'keep'이면 원본 크기 유지.",
    )
    group_tf.add_argument(
        "--min-sizes",
        type=int,
        nargs="+",
        default=[800],
        help="멀티스케일 예: 640 800 896 960 1024",
    )
    group_tf.add_argument(
        "--max-size",
        type=int,
        default=1333,
    )
    group_tf.add_argument(
        "--do-normalize",
        action="store_true",
        help="채널별 mean/std 정규화 사용",
    )
    group_tf.add_argument(
        "--normalize-mean",
        type=float,
        nargs="+",
        help="정규화 mean (길이 = in_channels)",
    )
    group_tf.add_argument(
        "--normalize-std",
        type=float,
        nargs="+",
        help="정규화 std (길이 = in_channels)",
    )
    group_tf.add_argument(
        "--rpn-small-anchors",
        action="store_true",
        help="resize-mode=keep 일 때 작은 앵커 사용",
    )
    return group_tf


def add_env_args(parser: argparse.ArgumentParser):
    group_env = parser.add_argument_group("Environment and Logging")
    group_env.add_argument(
        "--cuda",
        action="store_true",
        default=True,
        help="Enable CUDA training.",
    )
    group_env.add_argument(
        "--id-string",
        type=str,
        default="",
        help="Identifier string for the training run.",
    )
    group_env.add_argument(
        "--log-level",
        type=str,
        default="log",
        choices=["none", "log", "debug"],
        help="Logging level.",
    )
    group_env.add_argument(
        "--log-root",
        type=str,
        default=os.path.join(os.getcwd(), "models"),
        help="Root directory for saving models and logs.",
    )
    group_env.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() // 4,
        help="Number of CPU cores for data loading.",
    )
    group_env.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (if omitted, a random seed is generated and logged).",
    )
    return group_env


def add_kbrs_args(parser: argparse.ArgumentParser):
    group_kbrs = parser.add_argument_group("KBRS Specific")
    group_kbrs.add_argument(
        "--use-kbrs",
        action="store_true",
        help="Use KBRS loss during training.",
    )
    group_kbrs.add_argument(
        "--kbrs-param",
        action="append",
        metavar="KEY=VAL",
        help="Override KBRS_PARAMS entries, e.g., kernel_x=20",
    )
    group_kbrs.add_argument(
        "--loss-weights",
        nargs=2,
        action="append",
        metavar=("LOSS_NAME", "WEIGHT"),
        help="Set a weight for a specific loss. Repeatable.",
    )
    group_kbrs.add_argument(
        "--score-weights",
        nargs=2,
        action="append",
        metavar=("COMP_NAME", "WEIGHT"),
        help="KBRS scorer component weights (density/mixture/centeredness).",
    )
    return group_kbrs


def _apply_yaml_config(args):
    if not getattr(args, "config", None):
        return args

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if args.config_key is not None:
        if args.config_key not in cfg:
            raise ValueError(f"Config key '{args.config_key}' not found in {args.config}")
        cfg = cfg[args.config_key]
        
    # 1) 단순 스칼라/리스트 옵션: 키 이름이 args 속성과 같으면 덮어쓰기
    simple_keys = [
        "train_replays",
        "test_replays",
        "include_components",
        "label_method",
        "sample_ratio",
        "data_root",
        "interval",
        "val_count",
        "window_size",
        "batch_size",
        "learning_rate",
        "max_epoch",
        "test_eval_every",
        "do_inference_after_train",
        "resize_mode",
        "min_sizes",
        "max_size",
        "do_normalize",
        "normalize_mean",
        "normalize_std",
        "rpn_small_anchors",
        "cuda",
        "id_string",
        "log_level",
        "log_root",
        "num_workers",
        "seed",
        "use_kbrs",
    ]
    for k in simple_keys:
        if k in cfg and hasattr(args, k):
            val = cfg[k]
            # train/test_replays 는 [*set1, *set2] 같은 nested list 가 들어올 수 있으므로 flatten
            if k in ("train_replays", "test_replays"):
                val = _flatten_list(val)
            setattr(args, k, val)

    if "loss_weights" in cfg:
        current = list(getattr(args, "loss_weights", []) or [])

        if isinstance(cfg["loss_weights"], dict):
            # {'loss_objectness':1.0, ...} → [['loss_objectness','1.0'], ...]
            for name, w in cfg["loss_weights"].items():
                current.append([name, str(w)])
        else:
            # 이미 [['name', 'weight'], ...] 형태라면 그대로
            current.extend(cfg["loss_weights"])

        args.loss_weights = current

    # 3) score_weights: [["density", 0.3], ...] 형태를 쓰고 싶다면
    if "score_weights" in cfg:
        current = list(getattr(args, "score_weights", []) or [])
        current.extend(cfg["score_weights"])
        args.score_weights = current

    # 4) kbrs_params: nested dict → kbrs_param 리스트로 변환
    if "kbrs_params" in cfg:
        flat = list(_flatten_kbrs_params("", cfg["kbrs_params"]))
        current = list(getattr(args, "kbrs_param", []) or [])
        current.extend(flat)
        args.kbrs_param = current

    return args

# -------------------------
# Train
# -------------------------
def parse_train_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Minimal argument parser for Mask R-CNN training"
    )

    # ★ 여기서 config 옵션 하나 추가
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file to override/extend CLI arguments.",
    )
    
    parser.add_argument(
        "--config-key",
        type=str,
        default=None,
        help="Top-level key inside YAML to use as config (when YAML contains multiple presets).",
    )
    
    # Data and Labeling (train/test replays 분리)
    group_data = parser.add_argument_group("Data and Labeling")

    group_data.add_argument(
        "--train-replays",
        "--train-replay",
        type=str,
        nargs="+",
        help="Replay IDs used for training (one or more).",
    )
    group_data.add_argument(
        "--test-replays",
        "--test-replay",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional replay IDs used only for testing/validation. "
            "If omitted, validation/test will be drawn from train replays."
        ),
    )

    group_data.add_argument(
        "--label-method",
        type=str,
        default=config.LABEL_METHODS[0],
        choices=config.LABEL_METHODS,
        help="Label extraction method (folder name).",
    )
    group_data.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Fraction of training dataset to sample.",
    )
    group_data.add_argument(
        "--data-root",
        type=str,
        default=os.path.join(os.getcwd(), "data"),
        help="Root directory for data.",
    )
    group_data.add_argument(
        "--include-components",
        type=str,
        nargs="+",
        default=DEFAULT_COMPONENTS,
        help="List of components to include.",
    )
    group_data.add_argument(
        "--interval",
        type=int,
        default=config.INTERVAL,
        help="Sampling interval for frame windows (1 = use every index).",
    )
    group_data.add_argument(
        "--val-count",
        type=int,
        default=1000,
        help="Number of samples to use for validation (0 = no validation).",
    )

    # Model Hyperparameters
    group_hyper = parser.add_argument_group("Model Hyperparameters")
    group_hyper.add_argument(
        "--window-size",
        type=int,
        default=config.WINDOW_SIZE,
    )
    group_hyper.add_argument(
        "--batch-size",
        type=int,
        default=config.TRAIN_BATCH_SIZE,
    )
    group_hyper.add_argument(
        "--learning-rate",
        type=float,
        default=config.TRAIN_LEARNING_RATE,
    )
    group_hyper.add_argument(
        "--max-epoch",
        type=int,
        default=config.TRAIN_EPOCHS,
    )
    group_hyper.add_argument(
        "--test-eval-every",
        type=int,
        default=0,
        help="Test set evaluation interval in epochs (0 = disable).",
    )
    group_hyper.add_argument(
        "--do-inference-after-train",
        action="store_true",
        help="Run inference on the test set after training completes.",
    )

    # Transform / Env / KBRS 공통 헬퍼 사용
    add_transform_args(parser)
    add_env_args(parser)
    add_kbrs_args(parser)

    args = parser.parse_args(argv)
    args = _apply_yaml_config(args)
    
    if not args.train_replays:
        parser.error(
            "No train replays specified. "
            "Use --train-replays ... or provide 'train_replays' in --config (with optional --config-key)."
        )

    return args


# -------------------------
# Inference
# -------------------------
def build_inference_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Mask R-CNN inference on preprocessed StarCraft replays"
    )

    # replays + data-root + include-components
    add_data_args(parser, with_replays=True, with_label=False)

    group_data = parser.add_argument_group("Inference I/O")
    group_data.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.getcwd(), "predictions"),
        help="Directory to save prediction JSON files.",
    )
    group_data.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Subdirectory under --output-dir to save predictions.",
    )

    group_model = parser.add_argument_group("Model Loading")
    group_model.add_argument(
        "--model-root",
        type=str,
        default=os.path.join(os.getcwd(), "models"),
        help="Root directory for model checkpoints.",
    )
    group_model.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model folder to use.",
    )
    group_model.add_argument(
        "--model-number",
        type=int,
        required=True,
        help="Checkpoint number to use (e.g., 4 for model_4.pth).",
    )
    group_model.add_argument(
        "--label-method",
        type=str,
        default=config.LABEL_METHODS[0],
        choices=config.LABEL_METHODS,
        help="Label method for reference (not used in inference).",
    )
    group_model.add_argument(
        "--window-size",
        type=int,
        default=1,
        help="Window size for input frames, consistent with trained model.",
    )
    group_model.add_argument(
        "--use-kbrs",
        action="store_true",
        help="Use KBRS if available in the model.",
    )

    group_hyper = parser.add_argument_group("Inference Hyperparameters")
    group_hyper.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA if available.",
    )
    group_hyper.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    group_hyper.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
    )
    group_hyper.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Fraction of frames to sample for inference (0.0 < r <= 1.0).",
    )
    group_hyper.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() // 4,
        help="DataLoader workers.",
    )
    group_hyper.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling/order in inference (if omitted, a random seed is generated).",
    )

    return parser


def parse_inference_args(argv=None):
    return build_inference_parser().parse_args(argv)
