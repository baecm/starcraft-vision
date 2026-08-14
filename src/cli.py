# src/cli.py
import argparse
import os
import sys
import config

DEFAULT_COMPONENTS = ["worker", "ground", "air", "building", "vision"]


def add_data_args(
    parser: argparse.ArgumentParser,
    *,
    with_replays: bool = True,
    with_label: bool = False,
):
    """
    Common data argument group builder used for CLI interfaces.
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
            help="Label extraction method.",
        )
        group.add_argument(
            "--sample-ratio",
            type=float,
            default=1.0,
            help="Fraction of dataset to sample.",
        )

    return group


def build_inference_parser() -> argparse.ArgumentParser:
    """
    Build ArgumentParser for model inference.
    """
    parser = argparse.ArgumentParser(
        description="Run object detection inference on preprocessed StarCraft replays"
    )

    # Replays + Data Root + Components
    add_data_args(parser, with_replays=True, with_label=False)

    group_io = parser.add_argument_group("Inference I/O")
    group_io.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.getcwd(), "predictions"),
        help="Directory to save prediction JSON files.",
    )
    group_io.add_argument(
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
        help="Name of the model folder or architecture to use.",
    )
    group_model.add_argument(
        "--model-number",
        type=int,
        required=True,
        help="Checkpoint epoch number to use (e.g., 30 for model_030.pth).",
    )
    group_model.add_argument(
        "--label-method",
        type=str,
        default=config.LABEL_METHODS[0],
        choices=config.LABEL_METHODS,
        help="Label method for reference.",
    )
    group_model.add_argument(
        "--window-size",
        type=int,
        default=1,
        help="Window size for input frames.",
    )
    group_model.add_argument(
        "--use-kbrs",
        action="store_true",
        help="Use KBRS if available in the model.",
    )
    group_model.add_argument(
        "--rtdetr-version",
        type=str,
        default="v1",
        help="RT-DETR version (v1 or v2).",
    )
    group_model.add_argument(
        "--rtdetr-size",
        type=str,
        default="l",
        help="RT-DETR model size (s, m, l, x).",
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
        default=max(1, (os.cpu_count() or 4) // 4),
        help="DataLoader workers.",
    )
    group_hyper.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling/order in inference.",
    )

    return parser


def parse_inference_args(argv=None):
    """
    Parse CLI arguments for inference execution.
    """
    parser = build_inference_parser()
    args = parser.parse_args(argv)

    print("\n" + "=" * 70)
    print("Inference Arguments Configuration:")
    for key, value in vars(args).items():
        print(f"  - {key}: {value}")
    print("=" * 70 + "\n")

    return args
