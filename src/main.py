from training.train_loop import run_training
from config.base_parser import get_parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    print(f"[INFO] Training mode={args.mode}, num_classes={args.num_classes}, save_dir={args.log_save_dir}")
    run_training(args)
