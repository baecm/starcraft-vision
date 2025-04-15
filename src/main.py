import os
import argparse

from train import run_training

def parse_arguments():
    parser = argparse.ArgumentParser(description="Unified training entry for Mask R-CNN experiments")

    parser.add_argument("--train", "--training", action="store_true", dest="train", help="Training mode")
    parser.add_argument("--evaluate", "--eval", action="store_true", dest="evaluate", help="Evaluation only mode")
    
    parser.add_argument("--load-model", action="store_true", help="Whether to load pretrained model")
    parser.add_argument("--load-dir", type=str, help="Path to the dataset directory")
    parser.add_argument("--replays", type=int, nargs="+", help="Replay indices for training")

    parser.add_argument("--train-replays", type=int, nargs="+", default=None, help="Replay indices for training")
    parser.add_argument("--test-replays", type=int, nargs="+", default=None, help="Replay indices for testing")

    parser.add_argument("--cuda", type=bool, default=True, help="Use CUDA")
    parser.add_argument("--max-epoch", type=int, default=100, help="Max training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--window-size", type=int, default=1, help="Window size for temporal input (default: 1)")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate (default: 0.0001)")

    parser.add_argument("--log", action="store_true", help="Enable logging")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval (default: 10)")
    parser.add_argument("--log-root-dir", type=str, default=os.path.join(os.getcwd(), "models"), help="Directory to save models")
    parser.add_argument("--id-string", type=str, default="", help="Custom string for run identification")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    if args.train and args.evaluate:
        raise ValueError("Cannot specify both --train and --eval. Choose one.")
    
    print("\n[INFO] Arguments:")
    for arg, value in vars(args).items():
        print(f" - {arg}: {value}")
    
    if args.train:
        run_training(args)
        
    elif args.evaluate:
        pass
    

if __name__ == "__main__":
    main()
