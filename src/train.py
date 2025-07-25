import os
import argparse
import time
import torch
import tqdm
import utils
import json
import pickle
import wandb
from ultralytics import settings
from torch.utils.data import Subset

import detection.transforms as T
import config
from detection.engine import train_one_epoch, evaluate
from dataset.custom_penn_fudan import CustomPennFudanDataset
from model.maskrcnn_builder import get_model_instance_segmentation
from utils.logger import Logger


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
            Logger.info(f"[Preprocess] {msg}") if 'Logger' in globals() else print(f"[Preprocess] {msg}")

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

            # 누락된 필드 자동 보완
            coco.setdefault("info", {"description": "auto-generated", "version": "1.0"})
            coco.setdefault("licenses", [])
            coco.setdefault("categories", [{"id": 1, "name": "viewport"}])
            coco.setdefault("images", [])
            coco.setdefault("annotations", [])

            with open(pkl_path, "wb") as f:
                pickle.dump({
                    "info": coco["info"],
                    "licenses": coco["licenses"],
                    "categories": coco["categories"],
                    "images": coco["images"],
                    "annotations": coco["annotations"],
                }, f)

            log(f"{rid}: pickle created.")
        except Exception as e:
            log(f"{rid}: failed to process JSON: {e}")


def load_data(input_root, label_root, label_method, window_size, batch_size, replay_ids=None, train_replays=None, test_replays=None, test_size=50, sample_ratio=1.0, include_components=None):
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
    train_dataset = CustomPennFudanDataset(input_root, label_root, label_method, training_ids=train_ids, training=True, window_size=window_size, include_components=include_components)
    test_dataset = CustomPennFudanDataset(input_root, label_root, label_method, training_ids=test_ids, training=False, window_size=window_size, include_components=include_components)
    Logger.info(f"[Info] Full dataset size: Train {len(train_dataset)}, Test {len(test_dataset)}")
    
    # Apply sample ratio using torch.utils.data.Subset to avoid re-creating datasets
    if sample_ratio < 1.0:
        n_train = len(train_dataset)
        train_idx = torch.randperm(n_train).tolist()[:int(n_train * sample_ratio)]
        train_dataset = Subset(train_dataset, train_idx)
        
        n_test = len(test_dataset)
        test_idx = torch.randperm(n_test).tolist()[:int(n_test * sample_ratio)]
        test_dataset = Subset(test_dataset, test_idx)
        
        Logger.info(f"[Info] Applied sampling (ratio={sample_ratio}): Train {len(train_dataset)}, Test {len(test_dataset)}")

    train_loader = make_loader(train_dataset, batch_size, shuffle=True)
    test_loader = make_loader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader


def train_model(model, optimizer, lr_scheduler, data_loader_train, data_loader_test, device, num_epochs, save_dir, use_kbrs=False):
    Logger.info("[Stage] Starting training loop...")
    final_eval_stats = {}
    for epoch in tqdm.tqdm(range(num_epochs)):
        train_stats = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()
        eval_stats = evaluate(model, data_loader_test, device=device)

        # Log losses
        log_dict = {
            "Loss/train": train_stats.loss.global_avg,
            "Loss/class": train_stats.loss_classifier.global_avg,
            "Loss/box_reg": train_stats.loss_box_reg.global_avg,
            "Loss/mask": train_stats.loss_mask.global_avg,
            "Loss/objectness": train_stats.loss_objectness.global_avg,
            "Loss/rpn_box_reg": train_stats.loss_rpn_box_reg.global_avg,
        }

        if use_kbrs:
            if hasattr(train_stats, 'loss_kbrs'):
                log_dict["Loss/kbrs"] = train_stats.loss_kbrs.global_avg
            if hasattr(train_stats, 'loss_kbrs_density'):
                log_dict["Loss/kbrs_density"] = train_stats.loss_kbrs_density.global_avg
            if hasattr(train_stats, 'loss_kbrs_mixture'):
                log_dict["Loss/kbrs_mixture"] = train_stats.loss_kbrs_mixture.global_avg
            if hasattr(train_stats, 'loss_kbrs_centeredness'):
                log_dict["Loss/kbrs_centeredness"] = train_stats.loss_kbrs_centeredness.global_avg

        # Log evaluation stats
        # The evaluate function returns a CocoEvaluator object, from which we can extract stats.
        metric_dict = {}
        if hasattr(eval_stats, 'coco_eval'):
            stat_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
            for iou_type, coco_eval in eval_stats.coco_eval.items():
                for i, name in enumerate(stat_names):
                    metric_name = f"Eval/{iou_type}/{name}"
                    metric_value = coco_eval.stats[i]
                    metric_dict[metric_name] = metric_value
        ## Log metric_dict using Logger
        Logger.info(f"[Eval] Epoch {epoch}: {metric_dict}")
        log_dict.update(metric_dict)
        wandb.log(log_dict)
        
        final_eval_stats = metric_dict
        
        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_{epoch}.pth"))

    Logger.info("[Stage] Training complete!")
    return final_eval_stats


def run_training(args):
    settings.update({"wandb": True})
    Logger.info("[Stage] Preparing environment...")
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    Logger.info(f"[Info] Using device: {device}")

    if not args.id_string:
        id_str = f"{args.label_method}_win{args.window_size}_b{args.batch_size}"
        if args.use_kbrs:
            id_str += "_kbrs"
        args.id_string = id_str

    log_save_path = os.path.join(args.log_root, f"{args.id_string}_{time.strftime('%Y%m%d_%H%M%S')}/")
    os.makedirs(log_save_path, exist_ok=True)
    Logger.info(f"[Info] Log save path: {log_save_path}")

    # Initialize wandb
    wandb.init(project="starcraft", name=args.id_string, config=args)


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
        sample_ratio=args.sample_ratio,
        include_components=args.include_components
    )

    Logger.info("[Stage] Initializing model...")
    num_classes = 2  # background + viewport
    
    loss_weights = {}
    if args.loss_weight:
        for name, weight in args.loss_weight:
            loss_weights[name] = float(weight)

    kbrs_params = None
    if args.use_kbrs:
        if 'loss_kbrs' not in loss_weights and 'kbrs_loss_weight' in args and args.kbrs_loss_weight is not None:
            loss_weights['loss_kbrs'] = args.kbrs_loss_weight

        kbrs_params = {
            'weights': {"density": 1.0, "mixture": 0.7, "centeredness": 1.2},
            'region_size': config.KERNEL_SHAPE,
            'feature_map_name': '0',  # Use the first feature map from FPN
            'top_k_ratio': 0.5  # Use top 50% of GT boxes based on K-BRS score
        }

    train_ds = data_loader_train.dataset
    in_channels = len(train_ds.dataset.channel_indices if isinstance(train_ds, Subset) else train_ds.channel_indices)
    
    model = get_model_instance_segmentation(
        num_classes,
        window_size=args.window_size,
        in_channels=in_channels,
        do_normalize=False,
        use_kbrs=args.use_kbrs,
        kbrs_params=kbrs_params,
        loss_weights=loss_weights
    )
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_model(model, optimizer, lr_scheduler, data_loader_train, data_loader_test, device, args.max_epoch, log_save_path, use_kbrs=args.use_kbrs)

    # Finish wandb run
    wandb.finish()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Minimal argument parser for Mask R-CNN training")
    
    # Data and Labeling
    group_data = parser.add_argument_group("Data and Labeling")
    group_data.add_argument("--replays", type=str, nargs="+", required=True, help="List of replay IDs to include in dataset.")
    group_data.add_argument("--train-replays", type=str, nargs="+", default=None, help="Subset of replay IDs to use for training.")
    group_data.add_argument("--test-replays", type=str, nargs="+", default=None, help="Subset of replay IDs to use for testing.")
    group_data.add_argument("--label-method", type=str, default=config.LABEL_METHODS[0], choices=config.LABEL_METHODS, help="Label extraction method (folder name).")
    group_data.add_argument("--sample-ratio", type=float, default=1.0, help="Fraction of dataset to sample.")
    group_data.add_argument("--data-root", type=str, default=os.path.join(os.getcwd(), "data"), help="Root directory for data.")
    group_data.add_argument("--include-components", type=str, nargs='+', default=['worker', 'ground', 'air', 'building', 'vision'], help="List of components to include.")

    # Model Hyperparameters
    group_hyper = parser.add_argument_group("Model Hyperparameters")
    group_hyper.add_argument("--window-size", type=int, default=config.WINDOW_SIZE, help="Window size for input features.")
    group_hyper.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE, help="Batch size for training.")
    group_hyper.add_argument("--learning-rate", type=float, default=config.TRAIN_LEARNING_RATE, help="Initial learning rate.")
    group_hyper.add_argument("--max-epoch", type=int, default=config.TRAIN_EPOCHS, help="Maximum number of training epochs.")
    
    # KBRS Specific
    group_kbrs = parser.add_argument_group("KBRS Specific")
    group_kbrs.add_argument("--use-kbrs", action='store_true', help="Use KBRS loss during training.")
    group_kbrs.add_argument('--loss-weight', nargs=2, action='append', metavar=('LOSS_NAME', 'WEIGHT'), help='Set a weight for a specific loss. Can be used multiple times.')

    # Environment and Logging
    group_env = parser.add_argument_group("Environment and Logging")
    group_env.add_argument("--cuda", action='store_true', default=True, help="Enable CUDA training.")
    group_env.add_argument("--id-string", type=str, default="", help="Identifier string for the training run.")
    group_env.add_argument("--log-level", type=str, default="log", choices=["none", "log", "debug"], help="Logging level.")
    group_env.add_argument("--log-root", type=str, default=os.path.join(os.getcwd(), "models"), help="Root directory for saving models and logs.")
    
    return parser.parse_args()


if __name__ == "__main__":
    Logger.info("[Entry] Starting training script...")
    args = parse_arguments()
    Logger.set_level(args.log_level)
    run_training(args)
