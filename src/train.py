# src/train.py
import os
import subprocess
import time
import tqdm

import random
import secrets
import numpy as np

from types import SimpleNamespace

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from typing import Optional

import torch
from torch.utils.data import Subset

import wandb
from ultralytics import settings

import config

import detection.transforms as T
from detection.engine_safe import train_one_epoch_safe as train_one_epoch

from dataset.label_cache import ensure_label_pickles
from dataset.loader import load_data, make_loader
from dataset.custom_penn_fudan import CustomPennFudanDataset

from models.factory import build_model


from utils.logger import Logger
from utils.synology_chat import send_message
from utils.seed import set_global_seed

def _is_kbrs_enabled(cfg) -> bool:
    kbrs = getattr(cfg, "kbrs", None)
    if isinstance(kbrs, str):
        return kbrs.lower() == "enabled"
    return bool(getattr(cfg, "use_kbrs", False))


def _get_choice(group: str) -> Optional[str]:
    """
    Hydra가 현재 job에서 선택한 config group의 이름을 가져온다.
    예: group="dataset" -> "fold1"
        group="model"   -> "kbrs"
    """
    try:
        hc = HydraConfig.get()
        # hc.runtime.choices 는 dict: {"dataset": "fold1", "model": "kbrs", ...}
        return hc.runtime.choices.get(group)
    except Exception as e:
        Logger.warning(f"[_get_choice] failed for group={group}: {e}")
        return None


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
    id_string: str = "",
):
    Logger.info("[Stage] Starting training loop...")
    for epoch in tqdm.tqdm(range(num_epochs)):
        epoch_t0 = time.time()

        # ---- train ----
        t0 = time.time()
        # --- NaN 회피 옵션 (cfg에 없으면 안전한 기본값 사용) ---
        nan_log_path = os.path.join(save_dir, "nan_batches.jsonl")
        grad_clip_norm = 0.0
        lr_backoff = 0.5
        max_consecutive_nan = 20
        retry_fp32_on_nan = True

        # hydra cfg가 있는 경우 덮어쓰기(없어도 동작)
        try:
            grad_clip_norm = float(getattr(config, "GRAD_CLIP_NORM", grad_clip_norm))
        except Exception:
            pass

        train_stats = train_one_epoch(
            model,
            optimizer,
            data_loader_train,
            device,
            epoch,
            print_freq=10,
            scaler=None,
            nan_log_path=nan_log_path,
            skip_nonfinite=True,
            max_consecutive_nan=max_consecutive_nan,
            grad_clip_norm=grad_clip_norm,
            lr_backoff=lr_backoff,
            retry_fp32_on_nan=retry_fp32_on_nan,
        )
        t_train = time.time() - t0
        Logger.info(f"[Time][epoch {epoch}] train_one_epoch: {t_train:.1f}s")

        lr_scheduler.step()

        # 로그
        log_dict = {
            "epoch": epoch,
            "Loss/train": train_stats.loss.global_avg,
        }
        
        for k in ("skipped", "nan", "grad_nonfinite", "grad_norm"):
            m = getattr(train_stats, k, None)
            if m is not None and hasattr(m, "global_avg"):
                log_dict[f"Train/{k}"] = float(m.global_avg)

        meters = getattr(train_stats, "meters", {})
        for k, meter in meters.items():
            # 'loss_xxx' 형태의 모든 지표를 자동으로 추출하여 기록
            if k.startswith("loss_") and hasattr(meter, "global_avg"):
                key_name = k[5:]  # 'loss_classifier' -> 'classifier'
                log_dict[f"Loss/{key_name}"] = float(meter.global_avg)

        # # ---- validation (매 epoch) ----
        # eval_stats = None
        # t_eval = 0.0
        # if data_loader_validation is not None:
        #     t1 = time.time()
        #     eval_stats = evaluate(
        #         model,
        #         data_loader_validation,
        #         device=device,
        #         epoch=epoch,
        #     )
        #     t_eval = time.time() - t1
        #     Logger.info(f"[Time][epoch {epoch}] evaluate(val): {t_eval:.1f}s")

        # # ---- Validation IC metrics 로깅 ----
        # if eval_stats is not None and hasattr(eval_stats, "aggregates"):
        #     agg = eval_stats.aggregates
        #     for key in ["ic@000", "ic@030", "ic@050", "ic_multi", "ic_ratio"]:
        #         if key in agg:
        #             log_dict[f"Eval/{key}"] = float(agg[key])

        # # ---- Test set 평가 (N epoch마다, 전체 test set) ----
        # t_test_eval = 0.0
        # if (
        #     data_loader_test is not None
        #     and test_eval_every > 0
        #     and (epoch + 1) % test_eval_every == 0
        # ):
        #     Logger.info(
        #         f"[Stage] Test evaluation at epoch {epoch+1} "
        #         f"(every {test_eval_every} epochs)"
        #     )
        #     t_te0 = time.time()
        #     test_stats = evaluate(
        #         model,
        #         data_loader_test,
        #         device=device,
        #         epoch=epoch,
        #     )
        #     t_test_eval = time.time() - t_te0
        #     Logger.info(f"[Time][epoch {epoch}] evaluate(test): {t_test_eval:.1f}s")

        #     if hasattr(test_stats, "aggregates"):
        #         t_agg = test_stats.aggregates
        #         for key in ["ic@000", "ic@030", "ic@050", "ic_multi", "ic_ratio"]:
        #             if key in t_agg:
        #                 log_dict[f"Test/{key}"] = float(t_agg[key])

        # # ---- KBRS epoch-level 통계 & 이미지 로그 ----
        # t_kbrs = 0.0
        # t_kbrs_img = 0.0
        # if hasattr(model, "consume_epoch_kbrs"):
        #     tk0 = time.time()
        #     scalars, cache = model.consume_epoch_kbrs()
        #     t_kbrs = time.time() - tk0

        #     if scalars:
        #         scalars = {**{k: v for k, v in scalars.items()}, "epoch": epoch}
        #         log_dict.update(scalars)

        #     if cache is not None:
        #         ti0 = time.time()

        #         def _minmax01(t, eps=1e-6):
        #             t = t.float()
        #             mn = t.amin(dim=(-2, -1), keepdim=True)
        #             mx = t.amax(dim=(-2, -1), keepdim=True)
        #             return (t - mn) / (mx - eps + 1e-12)

        #         def _to_rgb(gray01):
        #             return gray01.expand(3, -1, -1)

        #         def _to_wandb_image(t3hw):
        #             return t3hw.permute(1, 2, 0).clamp(0, 1).cpu().numpy()

        #         total = cache["score_total"]
        #         comps = cache["comp_maps"]

        #         wandb.log(
        #             {
        #                 "epoch": epoch,
        #                 "kbrs_epoch/total": wandb.Image(
        #                     _to_wandb_image(_to_rgb(_minmax01(total)))
        #                 ),
        #             },
        #             commit=False,
        #         )

        #         for name, m in comps.items():
        #             wandb.log(
        #                 {
        #                     "epoch": epoch,
        #                     f"kbrs_epoch/{name}": wandb.Image(
        #                         _to_wandb_image(_to_rgb(_minmax01(m)))
        #                     ),
        #                 },
        #                 commit=False,
        #             )

        #         t_kbrs_img = time.time() - ti0

        # ---- wandb 스칼라 로그 ----
        t_wandb = time.time()
        wandb.log(log_dict, commit=True)
        t_wandb = time.time() - t_wandb
        Logger.info(f"[Time][epoch {epoch}] wandb.log (scalars): {t_wandb:.3f}s")

        # ---- 체크포인트 저장 ----
        t_ckpt = 0.0
        if (epoch + 1) % 1 == 0 or (epoch + 1) == num_epochs:
            tc0 = time.time()
            save_path = os.path.join(save_dir, f"model_{epoch+1:03d}.pth")
            torch.save(model.state_dict(), save_path)
            t_ckpt = time.time() - tc0
            Logger.info(
                f"[Info] Saved model checkpoint: {save_path} "
                f"(time: {t_ckpt:.2f}s)"
            )

        # ---- 슬랙/시놀로지 알림 ----
        t_msg = time.time()
        try:
            send_message(f"[{id_string}] Epoch {epoch+1} completed.")
        except Exception as e:
            Logger.error(f"Failed to send message: {e}")
        t_msg = time.time() - t_msg

        # # ---- epoch 전체 시간 요약 ----
        # epoch_time = time.time() - epoch_t0
        # Logger.info(
        #     "[Time][epoch {e}] summary: "
        #     "train={tr:.1f}s, val={ev:.1f}s, test={te:.1f}s, "
        #     "kbrs_scalar={kb:.3f}s, kbrs_img={kbi:.3f}s, "
        #     "wandb={wb:.3f}s, ckpt={ck:.2f}s, msg={msg:.2f}s, "
        #     "total={tot:.1f}s".format(
        #         e=epoch,
        #         tr=t_train,
        #         ev=t_eval,
        #         te=t_test_eval,
        #         kb=t_kbrs,
        #         kbi=t_kbrs_img,
        #         wb=t_wandb,
        #         ck=t_ckpt,
        #         msg=t_msg,
        #         tot=epoch_time,
        #     )
        # )

            
            
def _unwrap_subset(ds):
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds

def run_training(cfg: DictConfig):
    """
    Hydra DictConfig를 받아서 학습 전체를 수행.
    (예전 argparse-style args를 완전히 대체)
    """
    settings.update({"wandb": True})
    Logger.info("[Stage] Preparing environment...]")

    # 1) seed 처리 (필요하면 여기서 generate + set)
    seed = cfg.seed
    if seed is None:
        generated = secrets.randbits(31)
        seed = generated
        cfg.seed = generated  # DictConfig에 써줘도 됨 (struct=False 가정)
        Logger.info(f"[Seed] No seed provided in config; generated seed={generated}")
    else:
        Logger.info(f"[Seed] Using seed={seed}")
    set_global_seed(int(seed))

    # 2) 디바이스
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
    Logger.info(f"[Info] Using device: {device} (torch.cuda.is_available(): {torch.cuda.is_available()} / cfg.cuda: {cfg.cuda})")

    run_tags = []
    # 3) id_string / tag_string
    if not cfg.id_string:
        run_tags.append(cfg.architecture.model_name)         # maskrcnn or rtdetr
        Logger.info(f"[Info] Architecture: {cfg.architecture.model_name}")
        use_kbrs = _is_kbrs_enabled(cfg)
        run_tags.append("kbrs" if use_kbrs else "vanilla")   # kbrs enabled/disabled
        Logger.info(f"[Info] Using KBRS: {use_kbrs}")
        run_tags.append(f"win{cfg.window_size}")  # win4
        Logger.info(f"[Info] Window size: {cfg.window_size}")

        # hydra runtime choices에서 현재 job의 선택값을 읽어온다.
        dataset_name    = _get_choice("dataset")     # fold1
        Logger.info(f"[Info] Dataset choice: {dataset_name}")
        run_tags.append(f"{dataset_name}")
        
        seed_choice     = _get_choice("seed")        # s123 같은 group 이름 (있으면)
        Logger.info(f"[Info] Seed choice from Hydra: {seed_choice}")
        run_tags.append(f"s{seed_choice}" if seed_choice else f"s{cfg.seed}")  # seed123 (실제 값)
        
        # kbrs가 켜져 있을 때만 loss/score suffix 달기
        if use_kbrs:
            kbrs_weight  = _get_choice("kbrs_loss")   # kbrs025 ...
            if kbrs_weight:
                Logger.info(f"[Info] KBRS loss choice: {kbrs_weight}")
                run_tags.append(f"{kbrs_weight}" if kbrs_weight else None)
            kbrs_score_name = _get_choice("kbrs_score")  # base, density020 ...
            if kbrs_score_name:
                Logger.info(f"[Info] KBRS score choice: {kbrs_score_name}")
                run_tags.append(kbrs_score_name.replace("/", "_"))

        run_tags.append(f"{time.strftime('%Y%m%d_%H%M%S')}")

        cfg.id_string = "_".join(run_tags)
        Logger.info(f"[Info] Using id string: {cfg.id_string}")


    log_save_path = os.path.join(cfg.log_root, f"{cfg.id_string}/")
    os.makedirs(log_save_path, exist_ok=True)
    Logger.info(f"[Info] Log save path: {log_save_path}")

    if wandb.run is not None:
        wandb.finish()

    # Logger.info(f"CFG: \n{OmegaConf.to_yaml(cfg)}")

    # 4) W&B init (DictConfig → dict 변환)
    wandb.init(
        project="starcraft",
        name=cfg.id_string,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=run_tags,
    )

    # 5) 경로 설정
    input_root = os.path.join(cfg.data_root, "input/dst")
    label_root = os.path.join(cfg.data_root, "label/dst")

    # 6) 라벨 pickle 준비: train + test 전체
    train_replays = list(cfg.dataset.train_replays)
    test_replays = list(getattr(cfg.dataset, "test_replays", []) or [])
    all_replays = [str(r) for r in (train_replays + test_replays)]

    ensure_label_pickles(
        label_root=label_root,
        label_method=cfg.label_method,
        replay_ids=sorted(set(all_replays)),
        num_workers=cfg.num_workers,
    )
    Logger.info("[Info] JSON to Pickle conversion completed.")

    # 7) train + val 로더 (val은 test_replays에서 cfg.val_count 만큼)
    data_loader_train, data_loader_validation, inner_ds = load_data(
        input_root=input_root,
        label_root=label_root,
        label_method=cfg.label_method,
        window_size=cfg.window_size,
        interval=cfg.interval,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        train_replays=train_replays,
        val_replays=test_replays or None,
        sample_ratio=cfg.sample_ratio,
        include_components=list(cfg.include_components),
        val_count=cfg.val_count,
        seed=int(seed),
    )

    # 8) test 로더 (test_replays 전체)
    test_loader = None
    if test_replays:
        test_dataset = CustomPennFudanDataset(
            input_root,
            label_root,
            cfg.label_method,
            training_ids=[str(r) for r in test_replays],
            training=True,
            window_size=cfg.window_size,
            interval=cfg.interval,
            include_components=list(cfg.include_components),
            verbose=False,
        )
        test_loader = make_loader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )
        Logger.info(f"[Info] Test dataset size (full): {len(test_dataset)}")

    if data_loader_validation is not None:
        Logger.info(
            f"[Info] Data loaded: "
            f"Train {len(data_loader_train.dataset)}, "
            f"Validation {len(data_loader_validation.dataset)}"
        )
    else:
        Logger.info(
            f"[Info] Data loaded: Train {len(data_loader_train.dataset)}, Validation (none)"
        )

    Logger.info("[Stage] Initializing model...]")
    num_classes = 2  # background + viewport

    # --- (1) loss_weights: dict 로 가정 (Hydra config에서 설정) ---
    loss_weights = {}
    if "loss_weights" in cfg and cfg.loss_weights is not None:
        for name, weight in cfg.loss_weights.items():
            loss_weights[name] = float(weight)

    # --- (2) kbrs_params merge: 기본 KBRS_PARAMS 위에 config 덮어쓰기 ---
    kbrs_params = None
    if _is_kbrs_enabled(cfg):
        kbrs_params = OmegaConf.to_container(cfg.kbrs.kbrs_params, resolve=True)
        if "kbrs_params" in cfg and cfg.kbrs_params is not None:
            from omegaconf import DictConfig as DC
            if isinstance(cfg.kbrs_params, DC):
                extra = OmegaConf.to_container(cfg.kbrs_params, resolve=True)
            else:
                extra = dict(cfg.kbrs_params)
            kbrs_params.update(extra)

    # --- (3) 입력 채널 계산 ---
    train_ds = data_loader_train.dataset
    inner_ds = _unwrap_subset(train_ds)
    in_channels = len(inner_ds.channel_indices) * inner_ds.window_size
    Logger.info(
        f"[Info] Input channels: {in_channels} "
        f"(window size: {inner_ds.window_size})"
    )

    # model = get_model_instance_segmentation(
    #     num_classes=num_classes,
    #     window_size=cfg.window_size,
    #     in_channels=in_channels,
    #     do_normalize=cfg.do_normalize,
    #     normalize_mean=cfg.normalize_mean,
    #     normalize_std=cfg.normalize_std,
    #     resize_mode=cfg.resize_mode,
    #     min_sizes=cfg.min_sizes,
    #     max_size=cfg.max_size,
    #     rpn_small_anchors=cfg.rpn_small_anchors if cfg.resize_mode == "keep" else False,
    #     use_kbrs=cfg.kbrs.use_kbrs,
    #     kbrs_params=kbrs_params,
    #     loss_weights=loss_weights,
    # )
    
    # factory.py가 요구하는 인자들을 cfg에 병합(Fallback)해줍니다.
    # Hydra 설정에 없을 경우를 대비한 안전 장치입니다.
    OmegaConf.set_struct(cfg, False)

    cfg.model_name = getattr(cfg.architecture, "model_name", "maskrcnn")
    cfg.rtdetr_version = getattr(cfg.architecture, "rtdetr_version", "v2")
    cfg.rtdetr_size = getattr(cfg.architecture, "rtdetr_size", "l")
    cfg.use_density_peak = getattr(cfg.architecture, "use_density_peak", getattr(cfg, "use_density_peak", False))
    cfg.use_probabilistic_query = getattr(cfg.architecture, "use_probabilistic_query", getattr(cfg, "use_probabilistic_query", False))
    cfg.in_channels = in_channels
    cfg.num_classes = num_classes
    cfg.kbrs_params = kbrs_params
    cfg.loss_weights = loss_weights
    
    # RPN 스몰 앵커 옵션 (Mask R-CNN 전용)
    cfg.rpn_small_anchors = getattr(cfg, "rpn_small_anchors", False) if getattr(cfg, "resize_mode", "resize") == "keep" else False

    model = build_model(cfg)
    
    Logger.info(
        f"[Info] Model initialized with {num_classes} classes and "
        f"{in_channels} input channels.]"
    )
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=cfg.learning_rate, momentum=0.9, weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    test_eval_every = cfg.test_eval_every

    # === 실제 학습 ===
    train_model(
        model,
        optimizer,
        lr_scheduler,
        data_loader_train,
        data_loader_validation,
        device,
        cfg.max_epoch,
        log_save_path,
        use_kbrs=cfg.use_kbrs,
        data_loader_test=test_loader,
        test_eval_every=test_eval_every,
        id_string=cfg.id_string,
    )
    torch.cuda.empty_cache()
    send_message(f"@work Training run '{cfg.id_string}' completed successfully.")
    wandb.finish()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 디버깅용: 전체 config 출력
    print(OmegaConf.to_yaml(cfg))

    # 로그 레벨 설정
    Logger.set_level(cfg.log_level)

    try:
        Logger.info("[Entry] Starting training script...")
        run_training(cfg)
    except Exception as e:
        # id_string이 아직 비어있을 수 있으므로 안전하게 재구성
        if not cfg.id_string:
            id_str = f"{cfg.label_method}_win{cfg.window_size}_b{cfg.batch_size}"
            if _is_kbrs_enabled(cfg):
                id_str += "_kbrs"
            cfg.id_string = id_str

        error_message = f"Training run '{cfg.id_string}' failed with an error: {e}"
        Logger.error(error_message)
        try:
            send_message(f"@work " + error_message)
        except Exception as send_error:
            Logger.error(f"Failed to send error message: {send_error}")
        raise



if __name__ == "__main__":
    import sys
    from config import resolve_cli_aliases
    sys.argv[1:] = resolve_cli_aliases(sys.argv[1:])
    main()