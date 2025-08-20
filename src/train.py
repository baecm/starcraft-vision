import os
import argparse
import time
import tqdm
import utils
import json
import pickle
from multiprocessing import Pool

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision.utils import make_grid

import wandb
from ultralytics import settings

import detection.transforms as T
from detection.engine import train_one_epoch, evaluate
from dataset.custom_penn_fudan import CustomPennFudanDataset
from model.maskrcnn_builder import get_model_instance_segmentation

import config
from utils.logger import Logger
from utils.synology_chat import send_message


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def make_loader(ds, batch_size, shuffle, num_workers):
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=utils.collate_fn
    )

def _process_json_worker(args):
    """Helper function for parallel JSON processing."""
    rid, label_root, label_method = args
    json_path = os.path.join(label_root, f"{rid}.rep", f"{label_method}.json")
    pkl_path = os.path.join(label_root, f"{rid}.rep", f"{label_method}.pkl")

    if not os.path.exists(json_path):
        return f"Skipped {rid}: no JSON found."
    if os.path.exists(pkl_path):
        return f"Skipped {rid}: pickle already exists."

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

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
        return f"Success {rid}: pickle created."
    except Exception as e:
        return f"Failed {rid}: {e}"


def preprocess_json_to_pickle(label_root, label_method, replay_ids, num_workers, verbose=True):
    def log(msg):
        if verbose:
            Logger.info(f"[Preprocess] {msg}") if 'Logger' in globals() else print(f"[Preprocess] {msg}")

    log(f"Starting JSON to Pickle conversion for {len(replay_ids)} replays using {num_workers} workers.")
    
    tasks = [(rid, label_root, label_method) for rid in replay_ids]
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm.tqdm(pool.imap_unordered(_process_json_worker, tasks), total=len(tasks), desc="Preprocessing JSON to Pickle"))

    # Optional: Log summary
    success_count = sum(1 for r in results if r.startswith("Success"))
    skipped_exist_count = sum(1 for r in results if "pickle already exists" in r)
    skipped_no_json_count = sum(1 for r in results if "no JSON found" in r)
    failed_count = sum(1 for r in results if r.startswith("Failed"))

    log(f"Preprocessing complete. Success: {success_count}, Skipped (existing): {skipped_exist_count}, Skipped (no JSON): {skipped_no_json_count}, Failed: {failed_count}")
    
    if failed_count > 0:
        for r in results:
            if r.startswith("Failed"):
                log(r)


def load_data(input_root, label_root, label_method, window_size, batch_size, num_workers, replay_ids=None, train_replays=None, test_replays=None, test_size=50, sample_ratio=1.0, test_sample_ratio=1.0, include_components=None):
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

    # Apply sample ratio to the training dataset
    if sample_ratio < 1.0:
        n_train = len(train_dataset)
        train_idx = torch.randperm(n_train).tolist()[:int(n_train * sample_ratio)]
        train_dataset = Subset(train_dataset, train_idx)
        Logger.info(f"[Info] Applied sampling to train data (ratio={sample_ratio}): Train {len(train_dataset)}")

    # Apply sample ratio to the test dataset
    if test_sample_ratio < 1.0:
        n_test = len(test_dataset)
        test_idx = torch.randperm(n_test).tolist()[:int(n_test * test_sample_ratio)]
        test_dataset = Subset(test_dataset, test_idx)
        Logger.info(f"[Info] Applied sampling to test data (ratio={test_sample_ratio}): Test {len(test_dataset)}")

    # Limit test dataset size if test_size is provided
    if test_size > 0 and len(test_dataset) > test_size:
        Logger.info(f"[Info] Limiting test dataset from {len(test_dataset)} to {test_size} samples.")
        indices = torch.randperm(len(test_dataset)).tolist()[:test_size]
        test_dataset = Subset(test_dataset, indices)
        Logger.info(f"[Info] New test dataset size: {len(test_dataset)}")
    
    train_loader = make_loader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    test_loader = make_loader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def train_model(model, optimizer, lr_scheduler, data_loader_train, data_loader_test, device, num_epochs, save_dir, use_kbrs=False):
    Logger.info("[Stage] Starting training loop...")
    final_eval_stats = {}
    for epoch in tqdm.tqdm(range(num_epochs)):
        train_stats = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()
        eval_stats = evaluate(model, data_loader_test, device=device)

        # 1) 한 군데에서만 누적해서 로그할 딕셔너리 구성
        final_log = {
            "epoch": epoch,
            "Loss/train": train_stats.loss.global_avg,
            "Loss/class": train_stats.loss_classifier.global_avg,
            "Loss/box_reg": train_stats.loss_box_reg.global_avg,
            "Loss/mask": train_stats.loss_mask.global_avg,
            "Loss/objectness": train_stats.loss_objectness.global_avg,
            "Loss/rpn_box_reg": train_stats.loss_rpn_box_reg.global_avg,
        }

        if hasattr(eval_stats, 'coco_eval'):
            stat_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
            for iou_type, coco_eval in eval_stats.coco_eval.items():
                for i, name in enumerate(stat_names):
                    final_log[f"Eval/{iou_type}/{name}"] = coco_eval.stats[i]

        if use_kbrs:
            meters = getattr(train_stats, "meters", {})
            skip = {"loss_classifier","loss_box_reg","loss_mask","loss_objectness","loss_rpn_box_reg","loss"}
            for k, meter in meters.items():
                if k.startswith("loss_") and k not in skip and hasattr(meter, "global_avg"):
                    final_log[f"Loss/{k[5:]}"] = float(meter.global_avg)

        # 2) kbrs 스칼라/이미지 처리: 이미지 등 큰 객체는 별도 로그하되 같은 step으로 commit=False
        if hasattr(model, "consume_epoch_kbrs"):
            scalars, cache = model.consume_epoch_kbrs()
            if scalars:
                wandb.log(scalars | {"epoch": epoch}, step=epoch, commit=False)

            if cache is not None:
                def _minmax01(t, eps=1e-6):
                    t = t.float()
                    mn = t.amin(dim=(-2, -1), keepdim=True)
                    mx = t.amax(dim=(-2, -1), keepdim=True)
                    return (t - mn) / (mx - mn + eps)

                def _to_rgb(gray01):    # (1,H,W)->(3,H,W)
                    return gray01.expand(3, -1, -1)

                def _to_wandb_image(t3hw):
                    return t3hw.permute(1,2,0).clamp(0,1).cpu().numpy()

                total = cache["score_total"]
                comps = cache["comp_maps"]
                wandb.log({"kbrs_epoch/total": wandb.Image(_to_wandb_image(_to_rgb(_minmax01(total)))),
                        "epoch": epoch}, step=epoch, commit=False)
                for name, m in comps.items():
                    wandb.log({f"kbrs_epoch/{name}": wandb.Image(_to_wandb_image(_to_rgb(_minmax01(m)))),
                            "epoch": epoch}, step=epoch, commit=False)

        # 3) 마지막에 한 번만 commit (이 줄이 그 epoch의 유일 커밋)
        wandb.log(final_log, step=epoch)  # commit=True (기본값)
        
        if hasattr(model, "consume_epoch_kbrs"):
            scalars, cache = model.consume_epoch_kbrs()

            if scalars:
                wandb.log(scalars, step=epoch)

            if cache is not None:
                def _minmax01(t, eps=1e-6):
                    t = t.float()
                    mn = t.amin(dim=(-2, -1), keepdim=True)
                    mx = t.amax(dim=(-2, -1), keepdim=True)
                    return (t - mn) / (mx - mn + eps)

                def _to_rgb(gray01):    # (1,H,W)->(3,H,W)
                    return gray01.expand(3, -1, -1)

                def _to_wandb_image(t3hw):  # (3,H,W)->HWC
                    return t3hw.permute(1, 2, 0).clamp(0, 1).cpu().numpy()

                total = cache["score_total"]
                comps = cache["comp_maps"]

                wandb.log({"kbrs_epoch/total": wandb.Image(_to_wandb_image(_to_rgb(_minmax01(total))))}, step=epoch)

                for name, m in comps.items():
                    wandb.log({f"kbrs_epoch/{name}": wandb.Image(_to_wandb_image(_to_rgb(_minmax01(m))))}, step=epoch)

                # (선택) 그리드 요약
                keys = ["density", "mixture", "centeredness"]
                panels = [_to_rgb(_minmax01(total))] + [
                    _to_rgb(_minmax01(comps[k])) for k in keys if k in comps
                ]
                if panels:
                    grid = make_grid(panels, nrow=len(panels))
                    wandb.log({"kbrs_epoch/grid": wandb.Image(_to_wandb_image(grid))}, step=epoch)
        
        final_eval_stats = metric_dict
        
        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_{epoch}.pth"))
        try:
            send_message(f"Epoch {epoch} completed. Model saved at {os.path.join(save_dir, f'model_{epoch}.pth')}.")
        except Exception as e:
            Logger.error(f"Failed to send message: {e}")

    Logger.info("[Stage] Training complete!")
    return final_eval_stats


def run_training(args):
    settings.update({"wandb": True})
    # wandb.define_metric("epoch")
    # wandb.define_metric("*", step_metric="epoch")
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
    preprocess_json_to_pickle(label_root=label_root, label_method=args.label_method, replay_ids=args.replays, num_workers=args.num_workers)
    Logger.info("[Info] JSON to Pickle conversion completed.")
    # Load data
    data_loader_train, data_loader_test = load_data(
        input_root,
        label_root,
        args.label_method,
        args.window_size,
        args.batch_size,
        args.num_workers,
        replay_ids=args.replays,
        train_replays=args.train_replays,
        test_replays=args.test_replays,
        test_size=args.test_size,
        sample_ratio=args.sample_ratio,
        test_sample_ratio=args.test_sample_ratio,
        include_components=args.include_components
    )
    Logger.info(f"[Info] Data loaded: Train {len(data_loader_train.dataset)}, Test {len(data_loader_test.dataset)}")
    
    Logger.info("[Stage] Initializing model...")
    num_classes = 2  # background + viewport
    
    loss_weights = {}
    if args.loss_weight:
        for name, weight in args.loss_weight:
            loss_weights[name] = float(weight)
    Logger.info(f"[Info] Loss weights: {loss_weights}")
    kbrs_params = None
    if args.use_kbrs:
        if 'loss_kbrs' not in loss_weights and 'kbrs_loss_weight' in args and args.kbrs_loss_weight is not None:
            loss_weights['loss_kbrs'] = args.kbrs_loss_weight

        kbrs_params = config.KBRS_PARAMS.copy()
        Logger.info(f"[Info] Using KBRS parameters: {kbrs_params}")
        
    train_ds = data_loader_train.dataset
    inner_ds = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    in_channels = len(inner_ds.channel_indices) * inner_ds.window_size
    Logger.info(f"[Info] Input channels: {in_channels} (window size: {inner_ds.window_size})")
    model = get_model_instance_segmentation(
        num_classes,
        window_size=args.window_size,
        in_channels=in_channels,
        do_normalize=False,
        use_kbrs=args.use_kbrs,
        kbrs_params=kbrs_params,
        loss_weights=loss_weights
    )
    Logger.info(f"[Info] Model initialized with {num_classes} classes and {in_channels} input channels.")
    model.to(device)
    Logger.info(f"[Info] Model moved to device: {device}")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_model(model, optimizer, lr_scheduler, data_loader_train, data_loader_test, device, args.max_epoch, log_save_path, use_kbrs=args.use_kbrs)

    # Finish wandb run and send success notification
    wandb.finish()
    send_message(f"Training run '{args.id_string}' completed successfully.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Minimal argument parser for Mask R-CNN training")
    
    # Data and Labeling
    group_data = parser.add_argument_group("Data and Labeling")
    group_data.add_argument("--replays", type=str, nargs="+", required=True, help="List of replay IDs to include in dataset.")
    group_data.add_argument("--train-replays", type=str, nargs="+", default=None, help="Subset of replay IDs to use for training.")
    group_data.add_argument("--test-replays", type=str, nargs="+", default=None, help="Subset of replay IDs to use for testing.")
    group_data.add_argument("--label-method", type=str, default=config.LABEL_METHODS[0], choices=config.LABEL_METHODS, help="Label extraction method (folder name).")
    group_data.add_argument("--sample-ratio", type=float, default=1.0, help="Fraction of dataset to sample.")
    group_data.add_argument("--test-size", type=int, default=50, help="Maximum number of samples for the test set. Set to 0 to disable.")
    group_data.add_argument("--test-sample-ratio", type=float, default=0.05, help="Fraction of test dataset to sample.")
    group_data.add_argument("--data-root", type=str, default=os.path.join(os.getcwd(), "data"), help="Root directory for data.")
    group_data.add_argument("--include-components", type=str, nargs='+', default=['worker', 'ground', 'air', 'building', 'vision'], help="List of components to include.")

    # Model Hyperparameters
    group_hyper = parser.add_argument_group("Model Hyperparameters")
    group_hyper.add_argument("--window-size", type=int, default=config.WINDOW_SIZE, help="Window size for input features.")
    group_hyper.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE, help="Batch size for training.")
    group_hyper.add_argument("--learning-rate", type=float, default=config.TRAIN_LEARNING_RATE, help="Initial learning rate.")
    group_hyper.add_argument("--max-epoch", type=int, default=config.TRAIN_EPOCHS, help="Maximum number of training epochs.")
    
    # Environment and Logging
    group_env = parser.add_argument_group("Environment and Logging")
    group_env.add_argument("--cuda", action='store_true', default=True, help="Enable CUDA training.")
    group_env.add_argument("--id-string", type=str, default="", help="Identifier string for the training run.")
    group_env.add_argument("--log-level", type=str, default="log", choices=["none", "log", "debug"], help="Logging level.")
    group_env.add_argument("--log-root", type=str, default=os.path.join(os.getcwd(), "models"), help="Root directory for saving models and logs.")
    group_env.add_argument("--num-workers", type=int, default=os.cpu_count()//2, help="Number of CPU cores for data loading.")

    # KBRS Specific
    group_kbrs = parser.add_argument_group("KBRS Specific")
    group_kbrs.add_argument("--use-kbrs", action='store_true', help="Use KBRS loss during training.")
    group_kbrs.add_argument('--loss-weight', nargs=2, action='append', metavar=('LOSS_NAME', 'WEIGHT'), help='Set a weight for a specific loss. Can be used multiple times.')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    Logger.set_level(args.log_level)
    
    try:
        Logger.info("[Entry] Starting training script...")
        run_training(args)
    except Exception as e:
        # Ensure the id_string is available for the message
        if not args.id_string:
            id_str = f"{args.label_method}_win{args.window_size}_b{args.batch_size}"
            if args.use_kbrs:
                id_str += "_kbrs"
            args.id_string = id_str
            
        error_message = f"Training run '{args.id_string}' failed with an error: {e}"
        Logger.error(error_message)
        try:
            send_message(error_message)
        except Exception as send_error:
            Logger.error(f"Failed to send error message: {send_error}")
        raise  # Re-raise the exception after sending the notification
