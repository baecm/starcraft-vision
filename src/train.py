# src/train.py
import os
import subprocess
import time
import tqdm

import random
import secrets
import numpy as np

import torch
from torch.utils.data import Subset

import wandb
from ultralytics import settings

import config
from cli import parse_train_args

import detection.transforms as T
from detection.engine import train_one_epoch
from evaluate import evaluate

from dataset.label_cache import ensure_label_pickles
from dataset.loader import load_data, make_loader
from dataset.custom_penn_fudan import CustomPennFudanDataset

from model.maskrcnn_builder import get_model_instance_segmentation

from utils.logger import Logger
from utils.synology_chat import send_message


# def get_transform(train):
#     transforms = [T.ToTensor()]
#     if train:
#         transforms.append(T.RandomHorizontalFlip(0.5))
#     return T.Compose(transforms)


def set_global_seed(seed: int | None):
    """
    Python / NumPy / PyTorch (CPU/CUDA) 시드를 한 번에 설정.
    deterministic 옵션까지 켜서 최대한 재현성이 유지되게 함.
    """
    if seed is None:
        Logger.info("[Seed] No seed provided; running with default randomness.")
        return

    Logger.info(f"[Seed] Setting global seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 선택: 완전 deterministic 모드 (속도 약간 손해)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as e:
        Logger.warning(f"[Seed] Could not set cuDNN deterministic flags: {e}")


def train_model(
    model,
    optimizer,
    lr_scheduler,
    data_loader_train,
    data_loader_validation,
    device,
    num_epochs,
    save_dir,
    use_kbrs: bool = False,
    data_loader_test=None,
    test_eval_every: int = 0,
):
    Logger.info("[Stage] Starting training loop...")
    for epoch in tqdm.tqdm(range(num_epochs)):
        # ---- train ----
        train_stats = train_one_epoch(
            model, optimizer, data_loader_train, device, epoch, print_freq=10
        )
        lr_scheduler.step()

        # ---- validation (매 epoch) ----
        eval_stats = None
        if data_loader_validation is not None:
            eval_stats = evaluate(
                model,
                data_loader_validation,
                device=device,
                epoch=epoch,
            )

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
        
        # ---- Validation IC metrics 로깅 ----
        if eval_stats is not None and hasattr(eval_stats, "aggregates"):
            agg = eval_stats.aggregates
            for key in ["ic@000", "ic@030", "ic@050", "ic_multi", "ic_ratio"]:
                if key in agg:
                    log_dict[f"Eval/{key}"] = float(agg[key])

            for src_key, dst_name in [
                ("mean_density", "Eval/density"),
                ("mean_centeredness", "Eval/centeredness"),
                ("mean_mixture", "Eval/mixture"),
            ]:
                if src_key in agg:
                    log_dict[dst_name] = float(agg[src_key])

        # ---- Test set 평가 (N epoch마다, 전체 test set) ----
        if (
            data_loader_test is not None
            and test_eval_every > 0
            and (epoch + 1) % test_eval_every == 0
        ):
            Logger.info(
                f"[Stage] Test evaluation at epoch {epoch+1} (every {test_eval_every} epochs)"
            )
            test_stats = evaluate(
                model,
                data_loader_test,
                device=device,
                epoch=epoch,
            )
            if hasattr(test_stats, "aggregates"):
                t_agg = test_stats.aggregates
                for key in ["ic@000", "ic@030", "ic@050", "ic_multi", "ic_ratio"]:
                    if key in t_agg:
                        log_dict[f"Test/{key}"] = float(t_agg[key])
                for src_key, dst_name in [
                    ("mean_density", "Test/density"),
                    ("mean_centeredness", "Test/centeredness"),
                    ("mean_mixture", "Test/mixture"),
                ]:
                    if src_key in t_agg:
                        log_dict[dst_name] = float(t_agg[src_key])

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
    
    if getattr(args, "seed", None) is None:
        # 0 ~ 2^31-1 범위에서 하나 뽑기
        generated = secrets.randbits(31)
        args.seed = generated
        Logger.info(f"[Seed] No --seed provided; generated seed={generated}")
    else:
        Logger.info(f"[Seed] Using provided seed={args.seed}")
    set_global_seed(int(args.seed))
    
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

    # 1) 라벨 pickle 준비: train + test 전체
    all_replays = []
    if getattr(args, "train_replay", None):
        all_replays.extend(args.train_replay)
    if getattr(args, "test_replay", None):
        all_replays.extend(args.test_replay)

    ensure_label_pickles(
        label_root=label_root,
        label_method=args.label_method,
        replay_ids=sorted({str(r) for r in all_replays}),
        num_workers=args.num_workers,
    )
    Logger.info("[Info] JSON to Pickle conversion completed.")

    # 2) train + val (val은 test_replay에서 val_count만큼) 로더
    data_loader_train, data_loader_validation, inner_ds = load_data(
        input_root=input_root,
        label_root=label_root,
        label_method=args.label_method,
        window_size=args.window_size,
        interval=args.interval,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_replays=args.train_replay,
        val_replays=args.test_replay,
        sample_ratio=args.sample_ratio,
        include_components=args.include_components,
        val_count=args.val_count,
        seed=int(args.seed),
    )

    # 3) test 로더 (test_replay 전체)
    test_loader = None
    if getattr(args, "test_replay", None):
        test_ids = [str(r) for r in args.test_replay]
        test_dataset = CustomPennFudanDataset(
            input_root,
            label_root,
            args.label_method,
            training_ids=test_ids,
            training=True,
            window_size=args.window_size,
            interval=args.interval,
            include_components=args.include_components,
        )
        test_loader = make_loader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        Logger.info(f"[Info] Test dataset size (full): {len(test_dataset)}")

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

    test_eval_every = getattr(args, "test_eval_every", 0)

    train_model(
        model,
        optimizer,
        lr_scheduler,
        data_loader_train,
        data_loader_validation,
        device,
        args.max_epoch,
        log_save_path,
        use_kbrs=args.use_kbrs,
        data_loader_test=test_loader,
        test_eval_every=test_eval_every,
    )
    
    wandb.finish()
    send_message(f"@work Training run '{args.id_string}' completed successfully.")

    if getattr(args, "do_inference_after_train", False):
        last_epoch = args.max_epoch
        cmd = [
            "python",
            "-m",
            "inference",
            "--replays",
            *args.test_replay,             # test set 전체에 대해 inference
            "--model-root", args.log_root,
            "--model-name", tag_string,    # 또는 args.id_string 기준으로 조합
            "--model-number", f"{last_epoch}",
            "--data-root", args.data_root,
            "--label-method", args.label_method,
            "--window-size", str(args.window_size),
        ]
        if getattr(args, "seed", None) is not None:
            cmd.extend(["--seed", str(args.seed)])
        Logger.info(f"[Post-Train] Running inference: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    args = parse_train_args()
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
