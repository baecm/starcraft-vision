from training.train_loop import run_training

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Unified training entry for Mask R-CNN experiments")

    parser.add_argument("--log_save_dir", type=str, default="./saved_models/", help="Directory to save models")
    parser.add_argument("--load_model", type=bool, default=False, help="Whether to load pretrained model")
    parser.add_argument("--load_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--training", type=int, nargs="+", required=True, help="Training directory indices")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--window_size", type=int, default=4, help="Window size for temporal input")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--cuda", type=bool, default=True, help="Use CUDA")
    parser.add_argument("--cuda_idx", type=int, default=0, help="CUDA device index")
    parser.add_argument("--max_epoch", type=int, default=30, help="Max training epochs")

    parser.add_argument("--id_string", type=str, default="", help="Custom string for run identification")
    parser.add_argument("--eval", type=bool, default=False, help="Evaluation only mode")
    parser.add_argument("--mode", type=str, default="default", help="Dataset label mode: default/one/point/point2_labels/six")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes for segmentation")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    print(f"[INFO] Training mode={args.mode}, num_classes={args.num_classes}, save_dir={args.log_save_dir}")
    run_training(args)
