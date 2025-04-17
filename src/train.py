import os
import argparse
import time
import torch
import tqdm
import utils

from detection.engine import train_one_epoch, evaluate
from dataset.custom_penn_fudan import CustomPennFudanDataset
from model.maskrcnn_builder import get_model_instance_segmentation

import detection.transforms as T
from utils.logger import Logger

Logger.set_level("log")  # 로그 레벨: "none", "log", "debug"

def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def load_data(replays_dir, window_size, batch_size, train_replays=None, test_replays=None, test_size=50):
    Logger.info("[Stage] Loading data...")

    data_input_dst = os.path.join(args.data_root, "input", "dst")
    Logger.info(f"[Info] Loading replays from {data_input_dst}")

    replay_names = os.listdir(data_input_dst)
    replay_names = [r for r in replay_names if os.path.isdir(os.path.join(data_input_dst, r))]
    Logger.info(f"[Info] Found {len(replay_names)} valid replay directories")

    # 이제 제대로 분기
    if train_replays is not None:
        train_replays = list(map(str, train_replays))
        if test_replays is None:
            test_replays = list(sorted(set(replay_names) - set(train_replays)))
            Logger.log(f"[Auto] Using replays {test_replays} for testing")
        else:
            test_replays = list(map(str, test_replays))

        Logger.info(f"[Info] Train: {len(train_replays)} replays, Test: {len(test_replays)} replays")

        train_dataset = CustomPennFudanDataset(data_input_dst, training=train_replays, window_size=window_size)
        test_dataset = CustomPennFudanDataset(data_input_dst, training=test_replays, window_size=window_size)

    else:
        Logger.log("[Auto] No train-replays provided, falling back to random split")

        full_dataset = CustomPennFudanDataset(data_input_dst, training=replay_names, window_size=window_size)

        indices = torch.randperm(len(full_dataset)).tolist()
        train_indices, test_indices = indices[:-test_size], indices[-test_size:]

        Logger.info(f"[Info] Random split: Train {len(train_indices)}, Test {len(test_indices)}")

        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

        return make_loader(train_dataset, batch_size, shuffle=True), make_loader(test_dataset, batch_size=1, shuffle=False)

def make_loader(ds, batch_size, shuffle):
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=utils.collate_fn)

def train_model(model, optimizer, lr_scheduler, data_loader_train, data_loader_test, device, num_epochs, save_dir):
    Logger.info("[Stage] Starting training loop...")
    for epoch in tqdm.tqdm(range(num_epochs)):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)

        lr_scheduler.step()

        evaluate(model, data_loader_test, device=device)

        save_path = os.path.join(save_dir, f"model_{epoch}.pth")
        torch.save(model.state_dict(), save_path)

    Logger.info("[Stage] Training complete!")


def run_training(args):
    Logger.info("[Stage] Preparing environment...")
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    Logger.info(f"[Info] Using device: {device}")

    log_save_path = os.path.join(
        args.log_root,
        f"model_{args.id_string}_lr{args.learning_rate}_w_size{args.window_size}_{str(int(time.time()))[4:]}/"
    )
    os.makedirs(log_save_path, exist_ok=True)
    Logger.info(f"[Info] Log save path: {log_save_path}")

    data_loader_train, data_loader_test = load_data(
        args.replays, args.window_size, args.batch_size,
        train_replays=args.train_replays,
        test_replays=args.test_replays
    )

    Logger.info("[Stage] Initializing model...")
    num_classes = 2
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_model(model, optimizer, lr_scheduler, data_loader_train, data_loader_test, device, args.max_epoch, log_save_path)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Minimal argument parser for Mask R-CNN training")

    parser.add_argument("--replays", type=str, nargs="+", required=True, help="List of replay directories")
    parser.add_argument("--train-replays", type=int, nargs="+", default=None, help="Replay indices used for training")
    parser.add_argument("--test-replays", type=int, nargs="+", default=None, help="Replay indices used for testing")

    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--max-epoch", type=int, default=100)

    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--id-string", type=str, default="")
    parser.add_argument("--data-root", type=str, default=os.path.join(os.getcwd(), "data"))
    parser.add_argument("--log-root", type=str, default=os.path.join(os.getcwd(), "models"))

    return parser.parse_args()


if __name__ == "__main__":
    Logger.info("[Entry] Starting training script...")
    args = parse_arguments()
    run_training(args)
