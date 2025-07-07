import os
import argparse
import time
import torch
import tqdm
import utils
import json
import pickle

import detection.transforms as T
import config
from detection.engine import train_one_epoch, evaluate
from dataset.custom_penn_fudan import CustomPennFudanDataset
from model.maskrcnn_builder import get_model_instance_segmentation
from utils.logger import Logger
from torch.utils.tensorboard import SummaryWriter


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def make_loader(ds, batch_size, shuffle):
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

def preprocess_json_to_pickle(label_root, label_method, replay_ids, verbose=True):
    def log(msg):
        if verbose:
            Logger.info(f"[Preprocess] {msg}")

    for rid in replay_ids:
        json_path = os.path.join(label_root, f"{rid}.rep", f"{label_method}.json")
        pkl_path  = os.path.join(label_root, f"{rid}.rep", f"{label_method}.pkl")

        if not os.path.exists(json_path):
            log(f"Skipping {rid}: no JSON found.")
            continue
        if os.path.exists(pkl_path):
            log(f"{rid}: pickle already exists.")
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                coco = json.load(f)

            image_dict = {int(img["id"]): img for img in coco.get("images", [])}
            ann_dict = {}
            for ann in coco.get("annotations", []):
                image_id = int(ann["image_id"])
                ann_dict.setdefault(image_id, []).append(ann)

            with open(pkl_path, "wb") as f:
                pickle.dump({"images": image_dict, "annotations": ann_dict}, f)

            log(f"{rid}: pickle created.")
        except Exception as e:
            Logger.warn(f"{rid}: failed to process JSON: {e}")


def load_data(input_root, label_root, label_method, window_size, batch_size, replay_ids=None, train_replays=None, test_replays=None, test_size=50, sample_ratio=1.0):
    Logger.info("[Stage] Loading data...")
    Logger.info(f"[Info] Input root: {input_root}")
    Logger.info(f"[Info] Label root: {label_root}, method: {label_method}")

    # Determine replay ID list
    if replay_ids:
        all_ids = [str(r) for r in replay_ids]
        Logger.info(f"[Info] Using provided replay IDs: {all_ids}")
    else:
        all_ids = [d[:-4] for d in os.listdir(input_root) if d.endswith('.rep')]
        Logger.info(f"[Info] Found replay directories: {all_ids}")

    # Split into train and test sets
    if train_replays is not None:
        train_ids = [str(r) for r in train_replays]
        if test_replays is None:
            test_ids = sorted(set(all_ids) - set(train_ids))
            Logger.log(f"[Auto] Using replays {test_ids} for testing")
        else:
            test_ids = [str(r) for r in test_replays]
        Logger.info(f"[Info] Train IDs: {train_ids}, Test IDs: {test_ids}")
    else:
        # No explicit train/test split provided: use all replays for both training and testing
        train_ids = all_ids
        test_ids = all_ids
        Logger.info(f"[Info] No train-replays provided; using all replays for train and test: {all_ids}")

    # Build datasets
    train_dataset = CustomPennFudanDataset(input_root, label_root, label_method, training_ids=train_ids, window_size=window_size)
    test_dataset = CustomPennFudanDataset(input_root, label_root, label_method, training_ids=test_ids, window_size=window_size)
    
    # Apply sample ratio
    if sample_ratio < 1.0:
        n_train = len(train_dataset)
        n_test = len(test_dataset)
        train_idx = torch.randperm(n_train).tolist()[:int(n_train * sample_ratio)]
        test_idx = torch.randperm(n_test).tolist()[:int(n_test * sample_ratio)]
        train_dataset = CustomPennFudanDataset(input_root, label_root, label_method, training_ids=train_ids, window_size=window_size, indices=train_idx)
        test_dataset = CustomPennFudanDataset(input_root, label_root, label_method, training_ids=test_ids, window_size=window_size, indices=test_idx)
        Logger.info(f"[Info] Applied sampling: Train {len(train_dataset)}, Test {len(test_dataset)}")

    train_loader = make_loader(train_dataset, batch_size, shuffle=True)
    test_loader = make_loader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader


def train_model(model, optimizer, lr_scheduler, data_loader_train, data_loader_test, device, num_epochs, save_dir, writer):
    Logger.info("[Stage] Starting training loop...")
    for epoch in tqdm.tqdm(range(num_epochs)):
        train_stats = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()
        eval_stats = evaluate(model, data_loader_test, device=device)

        # Log losses
        writer.add_scalar("Loss/train", train_stats.loss.global_avg, epoch)
        writer.add_scalar("Loss/class", train_stats.loss_classifier.global_avg, epoch)
        writer.add_scalar("Loss/box_reg", train_stats.loss_box_reg.global_avg, epoch)
        writer.add_scalar("Loss/mask", train_stats.loss_mask.global_avg, epoch)
        writer.add_scalar("Loss/objectness", train_stats.loss_objectness.global_avg, epoch)
        writer.add_scalar("Loss/rpn_box_reg", train_stats.loss_rpn_box_reg.global_avg, epoch)

        if isinstance(eval_stats, dict):
            for k, v in eval_stats.items():
                writer.add_scalar(f"Eval/{k}", v, epoch)

        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_{epoch}.pth"))

    writer.close()
    Logger.info("[Stage] Training complete!")


def run_training(args):
    Logger.info("[Stage] Preparing environment...")
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    Logger.info(f"[Info] Using device: {device}")

    if not args.id_string:
        args.id_string = f"{args.label_method}_win{args.window_size}_b{args.batch_size}"

    log_save_path = os.path.join(args.log_root, f"{args.id_string}_{time.strftime('%Y%m%d_%H%M%S')}/")
    os.makedirs(log_save_path, exist_ok=True)
    Logger.info(f"[Info] Log save path: {log_save_path}")

    writer = SummaryWriter(log_dir=log_save_path)
    writer.add_hparams(
        {
            "replays": ", ".join(args.replays),
            "label_method": args.label_method,
            "sample_ratio": args.sample_ratio,
            "window_size": args.window_size,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_epoch": args.max_epoch
        },
        {}
    )
    

    # Define data roots
    input_root = os.path.join(args.data_root, "input/dst")
    label_root = os.path.join(args.data_root, "label/dst")
    
    # Convert JSON labels to pickle format
    preprocess_json_to_pickle(label_root=label_root, label_method=args.label_method, replay_ids=args.replays)

    # Load data
    data_loader_train, data_loader_test = load_data(
        input_root,
        label_root,
        args.label_method,
        args.window_size,
        args.batch_size,
        replay_ids=args.replays,
        train_replays=args.train_replays,
        test_replays=args.test_replays,
        sample_ratio=args.sample_ratio
    )

    Logger.info("[Stage] Initializing model...")
    num_classes = 2  # background + viewport
    model = get_model_instance_segmentation(num_classes, window_size=args.window_size, do_normalize=False)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_model(model, optimizer, lr_scheduler, data_loader_train, data_loader_test, device, args.max_epoch, log_save_path, writer)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Minimal argument parser for Mask R-CNN training"
    )
    parser.add_argument("--replays", type=str, nargs="+", required=True, help="List of replay IDs to include in dataset")
    parser.add_argument("--train-replays", type=str, nargs="+", default=None, help="Subset of replay IDs to use for training")
    parser.add_argument("--test-replays", type=str, nargs="+", default=None, help="Subset of replay IDs to use for testing")
    parser.add_argument("--label-method", type=str, default=config.LABEL_METHODS[0], choices=config.LABEL_METHODS, help="Label extraction method (folder name)")
    parser.add_argument("--sample-ratio", type=float, default=1.0, help="Fraction of dataset to sample")
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--cuda", action='store_true', default=True)
    parser.add_argument("--id-string", type=str, default="")
    parser.add_argument("--data-root", type=str, default=os.path.join(os.getcwd(), "data"))

    parser.add_argument("--log-level", type=str, default="log", choices=["none", "log", "debug"], help="Logging level")
    parser.add_argument("--log-root", type=str, default=os.path.join(os.getcwd(), "models"))
    return parser.parse_args()


if __name__ == "__main__":
    Logger.info("[Entry] Starting training script...")
    args = parse_arguments()
    Logger.set_level(args.log_level)
    run_training(args)
