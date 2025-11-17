# src/train.py
import os
import argparse
import time
import tqdm
import utils
import json, re, ast
import pickle
from multiprocessing import Pool

import torch
from torch.utils.data import Subset

import wandb
from ultralytics import settings

import detection.transforms as T
# from detection.engine import train_one_epoch, evaluate
from detection.engine import train_one_epoch
from evaluate import evaluate
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
    if ds is None:
        return None
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

    replay_ids = [str(r) for r in replay_ids]
    log(f"Starting JSON to Pickle conversion for {len(replay_ids)} replays using {num_workers} workers.")

    tasks = [(rid, label_root, label_method) for rid in replay_ids]

    with Pool(processes=num_workers) as pool:
        results = list(tqdm.tqdm(pool.imap_unordered(_process_json_worker, tasks),
                                 total=len(tasks),
                                 desc="Preprocessing JSON to Pickle"))

    success_count = sum(1 for r in results if r.startswith("Success"))
    skipped_exist_count = sum(1 for r in results if "pickle already exists" in r)
    skipped_no_json_count = sum(1 for r in results if "no JSON found" in r)
    failed_count = sum(1 for r in results if r.startswith("Failed"))

    log(f"Preprocessing complete. Success: {success_count}, "
        f"Skipped (existing): {skipped_exist_count}, "
        f"Skipped (no JSON): {skipped_no_json_count}, "
        f"Failed: {failed_count}")

    if failed_count > 0:
        for r in results:
            if r.startswith("Failed"):
                log(r)


def load_data(input_root, label_root, label_method, window_size, interval, batch_size,
              num_workers, replays, sample_ratio=1.0, include_components=None, val_count=1000):
    Logger.info("[Stage] Loading data...")
    Logger.info(f"[Info] Input root: {input_root}")
    Logger.info(f"[Info] Label root: {label_root}, method: {label_method}")

    # 필수: replays
    if not replays:
        raise ValueError("--replays 를 1개 이상 지정해야 합니다.")
    train_ids = [str(r) for r in replays]
    Logger.info(f"[Info] Train IDs: {train_ids}")

    # Build dataset (train only; test 분리 없음)
    train_dataset = CustomPennFudanDataset(
        input_root, label_root, label_method,
        training_ids=train_ids, training=True,
        window_size=window_size, interval=interval,
        include_components=include_components
    )
    Logger.info(f"[Info] Full dataset size: Train {len(train_dataset)}")
    Logger.info(f"[Info] Window size: {window_size}, Interval: {interval}")

    # ---- Train/Val split (원본 기준으로!) ----
    n_train = len(train_dataset)
    val_dataset = None
    if val_count and n_train > val_count:
        full_idx = torch.randperm(n_train).tolist()
        val_idx   = full_idx[-val_count:]
        train_idx = full_idx[:-val_count]

        train_dataset = Subset(train_dataset, train_idx)
        val_dataset   = Subset(train_dataset.dataset, val_idx)

    if sample_ratio < 1.0:
        n_train = len(train_dataset)
        keep = torch.randperm(n_train).tolist()[:int(n_train * sample_ratio)]
        train_dataset = Subset(train_dataset, keep)
        Logger.info(f"[Info] Applied sampling to train data (ratio={sample_ratio}): Train {len(train_dataset)}")

    train_loader = make_loader(train_dataset, batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = make_loader(val_dataset,   batch_size, shuffle=False, num_workers=num_workers) if val_dataset is not None else None
    return train_loader, val_loader


def train_model(model, optimizer, lr_scheduler, data_loader_train, data_loader_validation, device, num_epochs, save_dir, use_kbrs=False):
    Logger.info("[Stage] Starting training loop...")
    for epoch in tqdm.tqdm(range(num_epochs)):
        train_stats = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()

        eval_stats = evaluate(model, data_loader_validation, device=device) if data_loader_validation is not None else None

        log_dict = {
            "epoch": epoch,
            "Loss/train": train_stats.loss.global_avg,
            "Loss/class": train_stats.loss_classifier.global_avg,
            "Loss/box_reg": train_stats.loss_box_reg.global_avg,
            "Loss/mask": train_stats.loss_mask.global_avg,
            "Loss/objectness": train_stats.loss_objectness.global_avg,
            "Loss/rpn_box_reg": train_stats.loss_rpn_box_reg.global_avg,
        }

        if use_kbrs:
            meters = getattr(train_stats, "meters", {})
            skip = {"loss_classifier","loss_box_reg","loss_mask","loss_objectness","loss_rpn_box_reg","loss"}
            for k, meter in meters.items():
                if k.startswith("loss_") and k not in skip and hasattr(meter, "global_avg"):
                    log_dict[f"Loss/{k[5:]}"] = float(meter.global_avg)
        
        # === Observer Intersection / Kernel metrics 로깅 ===
        if eval_stats is not None and hasattr(eval_stats, "aggregates"):
            agg = eval_stats.aggregates

            # ic 계열
            for key in ["ic@000", "ic@030", "ic@050", "ic_multi", "ic_ratio"]:
                if key in agg:
                    log_dict[f"Eval/{key}"] = float(agg[key])

            # kernel 계열
            mapping = [
                ("mean_density", "Eval/density"),
                ("mean_centeredness", "Eval/centeredness"),
                ("mean_mixture", "Eval/mixture"),
            ]
            for src_key, dst_name in mapping:
                if src_key in agg:
                    log_dict[dst_name] = float(agg[src_key])

        if hasattr(model, "consume_epoch_kbrs"):
            scalars, cache = model.consume_epoch_kbrs()
            if scalars:
                scalars = {**{k: v for k, v in scalars.items()}, "epoch": epoch}
                log_dict.update(scalars)

            if cache is not None:
                def _minmax01(t, eps=1e-6):
                    t = t.float()
                    mn = t.amin(dim=(-2,-1), keepdim=True)
                    mx = t.amax(dim=(-2,-1), keepdim=True)
                    return (t - mn) / (mx - eps + 1e-12)
                def _to_rgb(gray01): return gray01.expand(3, -1, -1)
                def _to_wandb_image(t3hw): return t3hw.permute(1,2,0).clamp(0,1).cpu().numpy()

                total = cache["score_total"]
                comps = cache["comp_maps"]
                wandb.log({
                    "epoch": epoch,
                    "kbrs_epoch/total": wandb.Image(_to_wandb_image(_to_rgb(_minmax01(total))))
                }, commit=False)

                for name, m in comps.items():
                    wandb.log({
                        "epoch": epoch,
                        f"kbrs_epoch/{name}": wandb.Image(_to_wandb_image(_to_rgb(_minmax01(m))))
                    }, commit=False)

        wandb.log(log_dict, commit=True)

        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            save_path = os.path.join(save_dir, f"model_{epoch+1:03d}.pth")
            torch.save(model.state_dict(), save_path)
            Logger.info(f"[Info] Saved model checkpoint: {save_path}")

        try:
            send_message(f"Epoch {epoch+1} completed.")
        except Exception as e:
            Logger.error(f"Failed to send message: {e}")
            
            
def _unwrap_subset(ds):
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds

def run_training(args):
    settings.update({"wandb": True})
    Logger.info("[Stage] Preparing environment...]")
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    Logger.info(f"[Info] Using device: {device}")

    if not args.id_string:
        id_str = f"{args.label_method}_win{args.window_size}_b{args.batch_size}"
        if args.use_kbrs:
            id_str += "_kbrs"
        args.id_string = id_str

    tag_string = f"{args.id_string}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_save_path = os.path.join(args.log_root, f"{tag_string}/")
    os.makedirs(log_save_path, exist_ok=True)
    Logger.info(f"[Info] Log save path: {log_save_path}")

    wandb.init(project="starcraft", name=args.id_string, config=vars(args), tags=[tag_string])

    input_root = os.path.join(args.data_root, "input/dst")
    label_root = os.path.join(args.data_root, "label/dst")

    preprocess_json_to_pickle(
        label_root=label_root,
        label_method=args.label_method,
        replay_ids=args.replays,
        num_workers=args.num_workers
    )
    Logger.info("[Info] JSON to Pickle conversion completed.")

    data_loader_train, data_loader_validation = load_data(
        input_root,
        label_root,
        args.label_method,
        args.window_size,
        args.interval,
        args.batch_size,
        args.num_workers,
        replays=args.replays,
        sample_ratio=args.sample_ratio,
        include_components=args.include_components,
        val_count=args.val_count
    )

    if data_loader_validation is not None:
        Logger.info(f"[Info] Data loaded: Train {len(data_loader_train.dataset)}, "
                    f"Validation {len(data_loader_validation.dataset)}")
    else:
        Logger.info(f"[Info] Data loaded: Train {len(data_loader_train.dataset)}, Validation (none)")

    Logger.info("[Stage] Initializing model...]")
    num_classes = 2  # background + viewport

    # --- (1) loss_weights dict 구성 ---
    loss_weights = {}
    if args.loss_weights:
        for name, weight in args.loss_weights:
            loss_weights[name] = float(weight)

    # --- (2) score_weights dict 구성 (scorer 내부 비율) ---
    score_weights = {}
    if args.score_weights:
        for name, weight in args.score_weights:
            score_weights[name] = float(weight)

    # --- (3) kbrs_params merge ---
    kbrs_params = None
    if args.use_kbrs:
        kbrs_params = config.KBRS_PARAMS.copy()

        # merge CLI overrides
        if getattr(args, "kbrs_param", None):
            def _autocast(s):
                # try int -> float -> bool -> str
                if s.lower() in ("true", "false"):
                    return s.lower() == "true"
                try:
                    return int(s)
                except ValueError:
                    try:
                        return float(s)
                    except ValueError:
                        return s

            for item in args.kbrs_param:
                if "=" not in item:
                    Logger.warning(f"[KBRS] Skip invalid --kbrs-param: {item}")
                    continue
                k, v = item.split("=", 1)
                k, v = k.strip(), _autocast(v.strip())
                kbrs_params[k] = v

        Logger.info(f"[Info] Using KBRS parameters: {kbrs_params}")

    train_ds = data_loader_train.dataset
    inner_ds = _unwrap_subset(train_ds)
    in_channels = len(inner_ds.channel_indices) * inner_ds.window_size
    Logger.info(f"[Info] Input channels: {in_channels} (window size: {inner_ds.window_size})")

    model = get_model_instance_segmentation(
        num_classes=num_classes,
        window_size=args.window_size,
        in_channels=in_channels,
        do_normalize=args.do_normalize,
        normalize_mean=args.normalize_mean,
        normalize_std=args.normalize_std,
        resize_mode=args.resize_mode,
        min_sizes=args.min_sizes,
        max_size=args.max_size,
        rpn_small_anchors=args.rpn_small_anchors if args.resize_mode == "keep" else False,
        use_kbrs=args.use_kbrs,
        kbrs_params=kbrs_params,
        loss_weights=loss_weights,
    )
    Logger.info(f"[Info] Model initialized with {num_classes} classes and {in_channels} input channels.]")
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_model(model, optimizer, lr_scheduler, data_loader_train, data_loader_validation, device, args.max_epoch, log_save_path, use_kbrs=args.use_kbrs)

    wandb.finish()
    send_message(f"@work Training run '{args.id_string}' completed successfully.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Minimal argument parser for Mask R-CNN training")

    # Data and Labeling
    group_data = parser.add_argument_group("Data and Labeling")
    group_data.add_argument("--replays", type=str, nargs="+", required=True,
                            help="List of replay IDs to use (train set = whole set).")
    group_data.add_argument("--label-method", type=str, default=config.LABEL_METHODS[0],
                            choices=config.LABEL_METHODS, help="Label extraction method (folder name).")
    group_data.add_argument("--sample-ratio", type=float, default=1.0,
                            help="Fraction of training dataset to sample.")
    group_data.add_argument("--data-root", type=str, default=os.path.join(os.getcwd(), "data"),
                            help="Root directory for data.")
    group_data.add_argument("--include-components", type=str, nargs='+',
                            default=['worker', 'ground', 'air', 'building', 'vision'],
                            help="List of components to include.")
    group_data.add_argument("--interval", type=int, default=config.INTERVAL,
                            help="Sampling interval for frame windows (1 = use every index).")
    group_data.add_argument("--val-count", type=int, default=1000,
                            help="Number of samples to use for validation (0 = no validation).")

    # Model Hyperparameters
    group_hyper = parser.add_argument_group("Model Hyperparameters")
    group_hyper.add_argument("--window-size", type=int, default=config.WINDOW_SIZE)
    group_hyper.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    group_hyper.add_argument("--learning-rate", type=float, default=config.TRAIN_LEARNING_RATE)
    group_hyper.add_argument("--max-epoch", type=int, default=config.TRAIN_EPOCHS)

    # Transform / Resize / Normalize
    group_tf = parser.add_argument_group("Transform / Resize / Normalize")
    group_tf.add_argument("--resize-mode", type=str, choices=["resize", "keep"], default="resize", help="'resize'면 old 스타일(권장), 'keep'이면 원본 크기 유지.")
    group_tf.add_argument("--min-sizes", type=int, nargs="+", default=[800], help="멀티스케일 예: 640 800 896 960 1024 (resize-mode=resize 일 때만 의미)")
    group_tf.add_argument("--max-size", type=int, default=1333)
    group_tf.add_argument("--do-normalize", action="store_true", help="채널별 mean/std 정규화 사용")
    group_tf.add_argument("--normalize-mean", type=float, nargs="+", help="정규화 mean (길이 = in_channels)")
    group_tf.add_argument("--normalize-std", type=float, nargs="+", help="정규화 std (길이 = in_channels)")
    group_tf.add_argument("--rpn-small-anchors", action="store_true", help="resize-mode=keep 일 때 작은 앵커 사용")

    # Environment and Logging
    group_env = parser.add_argument_group("Environment and Logging")
    group_env.add_argument("--cuda", action='store_true', default=True, help="Enable CUDA training.")
    group_env.add_argument("--id-string", type=str, default="", help="Identifier string for the training run.")
    group_env.add_argument("--log-level", type=str, default="log", choices=["none", "log", "debug"], help="Logging level.")
    group_env.add_argument("--log-root", type=str, default=os.path.join(os.getcwd(), "models"), help="Root directory for saving models and logs.")
    group_env.add_argument("--num-workers", type=int, default=os.cpu_count()//4, help="Number of CPU cores for data loading.")

    # KBRS Specific
    group_kbrs = parser.add_argument_group("KBRS Specific")
    group_kbrs.add_argument("--use-kbrs", action='store_true', help="Use KBRS loss during training.")
    group_kbrs.add_argument("--kbrs-param", action="append", metavar="KEY=VAL", help="Override KBRS_PARAMS entries, e.g., --kbrs-param kernel_x=20 --kbrs-param kernel_y=12")
    group_kbrs.add_argument("--loss-weights", nargs=2, action='append', metavar=('LOSS_NAME', 'WEIGHT'), help="Set a weight for a specific loss. Can be used multiple times.")
    group_kbrs.add_argument('--score-weights', nargs=2, action='append', metavar=('COMP_NAME', 'WEIGHT'), help="KBRS scorer component weights (density/mixture/centeredness).")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    Logger.set_level(args.log_level)

    try:
        Logger.info("[Entry] Starting training script...")
        run_training(args)
    except Exception as e:
        if not args.id_string:
            id_str = f"{args.label_method}_win{args.window_size}_b{args.batch_size}"
            if args.use_kbrs:
                id_str += "_kbrs"
            args.id_string = id_str

        error_message = f"Training run '{args.id_string}' failed with an error: {e}"
        Logger.error(error_message)
        try:
            send_message(f"@work " + error_message)
        except Exception as send_error:
            Logger.error(f"Failed to send error message: {send_error}")
        raise
