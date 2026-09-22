# src/train.py
import json
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
from utils.torch_compat import disable_inductor

disable_inductor()

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


def _git_state() -> dict:
    """HEAD commit and whether the working tree is dirty, or why we cannot tell.

    The container only mounts src/, so there is no .git under /workspace and
    asking git directly fails with exit 128. The Makefile therefore reads the
    host checkout at launch and passes GIT_COMMIT / GIT_DIRTY through the
    environment; the subprocess path is the fallback for running outside the
    container.
    """
    env_commit = os.environ.get("GIT_COMMIT", "").strip()
    if env_commit:
        return {
            "commit": env_commit,
            "dirty": os.environ.get("GIT_DIRTY", "").strip() == "1",
            "source": "host env",
        }

    def _run(args):
        return subprocess.check_output(
            args, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        ).decode().strip()

    try:
        return {
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "subject": _run(["git", "log", "-1", "--format=%s"]),
            "committed_at": _run(["git", "log", "-1", "--format=%cI"]),
            "dirty": bool(_run(["git", "status", "--porcelain"])),
            "source": "subprocess",
        }
    except Exception as e:
        return {
            "commit": None,
            "error": f"{type(e).__name__}: {e}",
            "hint": "GIT_COMMIT was not in the environment and there is no .git here; "
                    "launch via the Makefile, which exports it from the host checkout.",
        }


def _model_family(model_name: str) -> str:
    """Coarse family for a model_name, so a whole line of work is filterable.

    `director_centernet` is a CenterNet variant, but filtering W&B on
    "centernet" would not match it, and the same applies to every
    deformable/video DETR name. The family tag sits alongside the exact
    model_name rather than replacing it. Order matters below: rtdetr is
    checked before the generic detr test.
    """
    name = (model_name or "").lower()
    if "maskrcnn" in name or "mask_rcnn" in name:
        return "maskrcnn"
    if "rtdetr" in name:
        return "rtdetr"
    if "centernet" in name or name.startswith("director"):
        return "centernet"
    if "detr" in name:
        return "detr"
    return name or "unknown"


def _build_run_tags(cfg) -> list:
    """Filterable W&B tags describing what actually varies between runs.

    Beyond the architecture/dataset/seed identity, this records the loss
    composition and the knobs that distinguish the current sweep, because a
    dozen runs whose names differ only by a suffix are unreadable in the UI
    otherwise. The commit is tagged too: an earlier round had ablations
    launched from checkouts at different commits, and that was only caught by
    checksumming prediction files afterwards.
    """
    arch = getattr(cfg, "architecture", None)
    model_name = str(getattr(arch, "model_name", "unknown"))
    family = _model_family(model_name)
    tags = [model_name] if model_name == family else [family, model_name]

    use_kbrs = _is_kbrs_enabled(cfg)
    tags.append("kbrs" if use_kbrs else "vanilla")
    tags.append(f"win{cfg.window_size}")
    tags.append(str(_get_choice("dataset") or "fold?"))
    tags.append(f"s{cfg.seed}")

    # Which of the optional objectives are actually on. This is the ablation
    # axis, and reading it off the run name is error-prone.
    weights = getattr(arch, "loss_weights", None) or {}
    active = [k for k in ("rmc", "rep", "sm") if float(weights.get(f"lambda_{k}", 0.0)) > 0.0]
    if "director" in model_name.lower():
        tags.append("L:" + ("+".join(active) if active else "hcm_only"))
        lam_sm = float(weights.get("lambda_sm", 0.0))
        if lam_sm > 0.0:
            tags.append(f"sm{lam_sm:g}")

    # Non-default knobs only, so the tag list stays short when nothing is swept.
    for key, default, fmt in (
        ("dense_positives", False, lambda v: "dense"),
        ("head_conv", config.DIRECTOR_HEAD_CONV, lambda v: f"hc{v}"),
        ("render_sigma", config.DIRECTOR_RENDER_SIGMA, lambda v: f"sig{v:g}"),
        ("trainable_layers", config.DIRECTOR_TRAINABLE_LAYERS, lambda v: f"tl{v}"),
        ("conf_threshold", config.DIRECTOR_TAU, lambda v: f"tau{v:g}"),
    ):
        value = getattr(arch, key, default)
        if value != default:
            tags.append(fmt(value))

    if use_kbrs:
        for group in ("kbrs_loss", "kbrs_score"):
            choice = _get_choice(group)
            if choice:
                tags.append(str(choice).replace("/", "_"))

    commit = _git_state().get("commit")
    if commit:
        tags.append(f"git:{commit[:9]}")

    return [t for t in tags if t]


def _write_run_provenance(save_dir: str, model, id_string: str) -> None:
    """Record which code and which knobs produced this run, next to the weights.

    Seven ablations were once launched across several machines whose checkouts
    were at different commits. Three of them silently reproduced the previous
    round byte for byte, and that was only caught afterwards by checksumming
    prediction files against the older run. A run that cannot state its own
    commit cannot be compared to another run, so record it before training
    rather than reconstructing it later.
    """
    base = getattr(model, "base_model", model)
    weights = getattr(base, "loss_weights", None)
    record = {
        "id_string": id_string,
        # Recorded because inference otherwise guesses the architecture by
        # substring-matching id_string, which silently yields "maskrcnn" for
        # any name that does not contain the model's own name.
        "model_class": type(base).__name__,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": _git_state(),
        "loss_weights": {k: float(v) for k, v in weights.items()} if weights else None,
    }
    for knob in (
        "down_ratio", "k_max", "conf_threshold", "render_sigma", "u_observers",
        "smooth_huber_delta", "smooth_warmup_start", "smooth_warmup_full",
        "soft_center_radius", "peak_border_margin", "trainable_layers",
        "head_conv", "dense_positives",
    ):
        if hasattr(base, knob):
            record[knob] = getattr(base, knob)

    git = record["git"]
    if git.get("commit"):
        Logger.info(
            f"[Provenance] commit {git['commit'][:9]}"
            f"{' (DIRTY)' if git.get('dirty') else ''}"
            f"{' - ' + git['subject'] if git.get('subject') else ''}"
            f" [{git.get('source')}]"
        )
        if git.get("dirty"):
            Logger.warn("[Provenance] working tree is dirty; this run is not reproducible from a commit.")
    else:
        Logger.warn(f"[Provenance] could not determine git commit: {git.get('error')}")
        Logger.warn(f"[Provenance] {git.get('hint', '')}")

    try:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "run_provenance.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, sort_keys=True)
    except Exception as e:
        Logger.warn(f"[Provenance] failed to write run_provenance.json: {e}")


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
    _write_run_provenance(save_dir, model, id_string)

    Logger.info("[Stage] Starting training loop...")
    for epoch in tqdm.tqdm(range(num_epochs)):
        epoch_t0 = time.time()

        # Loss terms that ramp over training (currently only Director's
        # L_smooth) read the epoch off the module. Walking .modules() rather
        # than calling model.set_epoch() keeps this working through
        # KBRSWrapper and any future wrapper without each one forwarding it.
        for _module in model.modules():
            if hasattr(_module, "set_epoch"):
                _module.set_epoch(epoch)

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

    # 3) run tags / id_string
    #
    # These are built unconditionally. They used to live inside the
    # `if not cfg.id_string` branch, so every run launched with an explicit
    # id_string - which is every ablation - reached wandb.init with tags=[].
    # The parameters were still in wandb's config, but nothing was filterable
    # in the UI, which is the part that makes a sweep of near-identical runs
    # readable.
    run_tags = _build_run_tags(cfg)

    if not cfg.id_string:
        cfg.id_string = "_".join(run_tags + [time.strftime("%Y%m%d_%H%M%S")])
        Logger.info(f"[Info] Using id string: {cfg.id_string}")
    Logger.info(f"[Info] W&B tags: {run_tags}")


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

    model_arch = str(getattr(cfg.architecture, "model_name", "")).lower()
    is_director = "director" in model_arch

    if is_director:
        from dataset.mode_cache import ensure_mode_cache
        ensure_mode_cache(
            label_root=label_root,
            label_method=cfg.label_method,
            replay_ids=sorted(set(all_replays)),
            sigma=float(getattr(cfg, "mode_extraction_sigma", config.MODE_EXTRACTION_SIGMA)),
            min_sep=float(getattr(cfg, "mode_extraction_min_sep", config.MODE_EXTRACTION_MIN_SEP)),
            rel_threshold=float(getattr(cfg, "mode_extraction_rel_threshold", config.MODE_EXTRACTION_REL_THRESHOLD)),
            max_modes=int(getattr(cfg, "mode_extraction_max_modes", config.MODE_EXTRACTION_MAX_MODES)),
            num_workers=cfg.num_workers,
        )

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
        pair_mode=is_director,
        use_mode_cache=is_director,
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

    # --- (1) loss_weights: dict 로 가정 (Hydra config 및 architecture yaml에서 설정) ---
    loss_weights = {}
    arch_loss_weights = getattr(cfg.architecture, "loss_weights", None)
    if arch_loss_weights is not None:
        for name, weight in arch_loss_weights.items():
            loss_weights[name] = float(weight)
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

    # build_model reads these off the top level, so an architecture yaml that
    # defines them is otherwise inert and the model silently falls back to the
    # src/config.py defaults.
    for _arch_key in (
        "centernet_down_ratio", "max_objs",
        "k_max", "conf_threshold", "render_sigma", "u_observers",
        "smooth_huber_delta", "smooth_warmup_start", "smooth_warmup_full",
        "soft_center_radius", "peak_border_margin", "trainable_layers",
        "head_conv", "dense_positives",
    ):
        if hasattr(cfg.architecture, _arch_key):
            setattr(cfg, _arch_key, getattr(cfg.architecture, _arch_key))
    
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