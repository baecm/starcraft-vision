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
import config

from torch.utils.tensorboard import SummaryWriter


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def load_data(replays_dir, label_method, window_size, batch_size, train_replays=None, test_replays=None, test_size=50, sample_ratio=1.0):
    Logger.info("[Stage] Loading data...")

    data_input_dst = os.path.join(args.data_root, "pair")
    Logger.info(f"[Info] Loading replays from {data_input_dst}")

    replay_names = os.listdir(data_input_dst)
    replay_names = [r for r in replay_names if os.path.isdir(os.path.join(data_input_dst, r))]
    Logger.info(f"[Info] Found {len(replay_names)} valid replay directories")

    if train_replays is not None:
        train_replays = list(map(str, train_replays))
        if test_replays is None:
            test_replays = list(sorted(set(replay_names) - set(train_replays)))
            Logger.log(f"[Auto] Using replays {test_replays} for testing")
        else:
            test_replays = list(map(str, test_replays))

        Logger.info(f"[Info] Train: {len(train_replays)} replays, Test: {len(test_replays)} replays")

        full_train_dataset = CustomPennFudanDataset(data_input_dst, label_method, training=train_replays, window_size=window_size)
        full_test_dataset = CustomPennFudanDataset(data_input_dst, label_method, training=test_replays, window_size=window_size)

        if sample_ratio < 1.0:
            train_sampled_indices = torch.randperm(len(full_train_dataset)).tolist()[:int(len(full_train_dataset) * sample_ratio)]
            test_sampled_indices = torch.randperm(len(full_test_dataset)).tolist()[:int(len(full_test_dataset) * sample_ratio)]

            train_dataset = CustomPennFudanDataset(
                data_input_dst, label_method, training=train_replays,
                window_size=window_size, indices=train_sampled_indices
            )
            test_dataset = CustomPennFudanDataset(
                data_input_dst, label_method, training=test_replays,
                window_size=window_size, indices=test_sampled_indices
            )

            Logger.info(f"[Info] Applied sampling: Train {len(train_dataset)}, Test {len(test_dataset)}")
        else:
            train_dataset = full_train_dataset
            test_dataset = full_test_dataset

    else:
        Logger.log("[Auto] No train-replays provided, falling back to random split")

        full_dataset = CustomPennFudanDataset(data_input_dst, label_method, training=replay_names, window_size=window_size)
        total_len = len(full_dataset)
        sample_len = int(total_len * sample_ratio)

        indices = torch.randperm(total_len).tolist()[:sample_len]
        test_len = min(test_size, sample_len // 5)
        train_indices, test_indices = indices[:-test_len], indices[-test_len:]

        Logger.info(f"[Info] Random split (sampled): Train {len(train_indices)}, Test {len(test_indices)}")

        train_dataset = CustomPennFudanDataset(
            data_input_dst, label_method, training=replay_names,
            window_size=window_size, indices=train_indices
        )
        test_dataset = CustomPennFudanDataset(
            data_input_dst, label_method, training=replay_names,
            window_size=window_size, indices=test_indices
        )

    return make_loader(train_dataset, batch_size, shuffle=True), make_loader(test_dataset, batch_size=1, shuffle=False)



def make_loader(ds, batch_size, shuffle):
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=utils.collate_fn)


def train_model(model, optimizer, lr_scheduler, data_loader_train, data_loader_test, device, num_epochs, save_dir, writer):
    Logger.info("[Stage] Starting training loop...")
    for epoch in tqdm.tqdm(range(num_epochs)):
        train_stats = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()

        eval_stats = evaluate(model, data_loader_test, device=device)

        # MetricLogger -> 각 loss 항목에 접근하려면 .loss.global_avg 식으로 접근해야 함
        writer.add_scalar("Loss/train", train_stats.loss.global_avg, epoch)
        writer.add_scalar("Loss/class", train_stats.loss_classifier.global_avg, epoch)
        writer.add_scalar("Loss/box_reg", train_stats.loss_box_reg.global_avg, epoch)
        writer.add_scalar("Loss/mask", train_stats.loss_mask.global_avg, epoch)
        writer.add_scalar("Loss/objectness", train_stats.loss_objectness.global_avg, epoch)
        writer.add_scalar("Loss/rpn_box_reg", train_stats.loss_rpn_box_reg.global_avg, epoch)

        # Eval 로그 (dict일 경우만 기록)
        if isinstance(eval_stats, dict):
            for k, v in eval_stats.items():
                writer.add_scalar(f"Eval/{k}", v, epoch)

        # 모델 저장
        save_path = os.path.join(save_dir, f"model_{epoch}.pth")
        torch.save(model.state_dict(), save_path)

    writer.close()
    Logger.info("[Stage] Training complete!")


def run_training(args):
    Logger.info("[Stage] Preparing environment...")
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    Logger.info(f"[Info] Using device: {device}")

    if not args.id_string:
        args.id_string = f"{args.label_method}_win{args.window_size}_b{args.batch_size}"

    log_save_path = os.path.join(
        args.log_root,
        f"{args.id_string}_{time.strftime('%Y%m%d_%H%M%S')}/"
    )
    os.makedirs(log_save_path, exist_ok=True)
    Logger.info(f"[Info] Log save path: {log_save_path}")

    writer = SummaryWriter(log_dir=log_save_path)

    writer.add_hparams(
        {
            "replays": ', '.join(args.replays),
            "label_method": args.label_method,
            "sample_ratio": args.sample_ratio,
            "window_size": args.window_size,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_epoch": args.max_epoch,
        },
        {          
        }
    )

    data_loader_train, data_loader_test = load_data(
        args.replays, args.label_method, args.window_size, args.batch_size,
        train_replays=args.train_replays,
        test_replays=args.test_replays,
        sample_ratio=args.sample_ratio
    )

    Logger.info("[Stage] Initializing model...")
    num_classes = 2
    model = get_model_instance_segmentation(num_classes, window_size=args.window_size)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_model(model, optimizer, lr_scheduler, data_loader_train, data_loader_test, device, args.max_epoch, log_save_path, writer)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Minimal argument parser for Mask R-CNN training")

    parser.add_argument("--replays", type=str, nargs="+", required=True, help="List of replay directories")
    parser.add_argument("--train-replays", type=int, nargs="+", default=None, help="Replay indices used for training")
    parser.add_argument("--test-replays", type=int, nargs="+", default=None, help="Replay indices used for testing")
    parser.add_argument("--label-method", type=str, default=config.LABEL_METHODS[0], choices=config.LABEL_METHODS, help="Label extraction method")
    parser.add_argument("--sample-ratio", type=float, default=1.0, help="Fraction of dataset to use (e.g., 0.1 for 10%)")

    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--max-epoch", type=int, default=100)

    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--id-string", type=str, default="")
    parser.add_argument("--data-root", type=str, default=os.path.join(os.getcwd(), "data"))
    parser.add_argument("--log-root", type=str, default=os.path.join(os.getcwd(), "models"))
    parser.add_argument("--log-level", type=str, default="log", choices=["none", "log", "debug"], help="Logging level")


    return parser.parse_args()


if __name__ == "__main__":
    Logger.info("[Entry] Starting training script...")
    args = parse_arguments()
    Logger.set_level(args.log_level)
    run_training(args)
