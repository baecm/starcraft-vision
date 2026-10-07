"""
Train one model from the Hydra config (conf/config.yaml).

    make train ARGS="architecture=director_centernet seed=456 id_string=..."

run_training does, in this order (the order fixes the random stream, so a
seed reproduces a run; do not reorder the steps):

  1. seed everything                      _seed_everything
  2. pick the device                      _select_device
  3. name the run, start W&B              _start_run
  4. build label / mode caches, loaders   _prepare_data
  5. build the model                      _build_model
  6. optimizer and LR schedule            _build_optimizer_and_schedule
  7. resume from a checkpoint, if asked   _maybe_resume
  8. train, checkpointing as it goes      train_model

Everything the model is built from goes through _model_args, the one list of
what build_model reads. Run identity (W&B tags, run_provenance.json next to
the weights) is recorded so that a run can state which code and which knobs
produced it.
"""
import json
import os
import secrets
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import hydra
import torch
import tqdm
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from ultralytics import settings

import config
from dataset.starcraft_windows import MODE_TARGET_CHOICES
from dataset.label_cache import ensure_label_pickles
from dataset.loader import load_data
from detection.engine_safe import train_one_epoch_safe as train_one_epoch
from models.factory import build_model
from utils.logger import Logger
from utils.seed import set_global_seed
from utils.synology_chat import send_message
from utils.torch_compat import disable_inductor

disable_inductor()

NUM_CLASSES = 2  # background + viewport
SGD_MOMENTUM = 0.9
SGD_WEIGHT_DECAY = 0.0005
# NaN handling in the training loop (detection/engine_safe.py)
LR_BACKOFF_ON_NAN = 0.5
MAX_CONSECUTIVE_NAN = 20
PRINT_FREQ = 10


# ---------------------------------------------------------------------------
# Reading the config
# ---------------------------------------------------------------------------

def _is_kbrs_enabled(cfg) -> bool:
    kbrs = getattr(cfg, "kbrs", None)
    if isinstance(kbrs, str):
        return kbrs.lower() == "enabled"
    return bool(getattr(cfg, "use_kbrs", False))


def _get_choice(group: str) -> Optional[str]:
    """The option Hydra chose for a config group in this job, e.g. "fold1" for "dataset"."""
    try:
        return HydraConfig.get().runtime.choices.get(group)
    except Exception as e:
        Logger.warn(f"[_get_choice] failed for group={group}: {e}")
        return None


def _mode_targets(cfg) -> str:
    """cfg.mode_targets, validated. Only Mask R-CNN reads it: the heatmap
    models take the ranked modes as their own targets already."""
    value = str(cfg.get("mode_targets", "none") or "none")
    if value not in MODE_TARGET_CHOICES:
        raise ValueError(f"mode_targets must be one of {MODE_TARGET_CHOICES}, got {value!r}")
    model_name = str(getattr(getattr(cfg, "architecture", None), "model_name", "")).lower()
    if value != "none" and model_name != "maskrcnn":
        raise ValueError(f"mode_targets={value!r} is implemented for maskrcnn only, not {model_name!r}")
    return value


# ---------------------------------------------------------------------------
# Run identity: W&B tags and run_provenance.json
# ---------------------------------------------------------------------------

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
    "centernet" would not match it. The family tag sits alongside the exact
    model_name rather than replacing it. (Older runs also used "rtdetr" and
    "detr", for the architectures now in archive/legacy/.)
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


def _host_tag() -> Optional[str]:
    """Which workstation and which physical GPU produced this run, e.g. worker07:0.

    Neither half is discoverable from inside the container. Its hostname is the
    container id, and NVIDIA_VISIBLE_DEVICES=2 exposes that one card as cuda:0,
    so torch.cuda reports device 0 whichever card is actually in use. The
    Makefile exports HOST_NAME for the same reason it exports GIT_COMMIT, and
    compose already forwards NVIDIA_VISIBLE_DEVICES.

    This matters beyond bookkeeping: a batch of ablations was once launched
    across machines whose checkouts were at different commits, and a later
    stride run came from a workstation that had not pulled, which made it a
    silently different experiment. Recording the machine puts that in the run
    itself rather than in someone's shell history.
    """
    host = (os.environ.get("HOST_NAME") or "").strip()
    if not host:
        return None
    devices = (os.environ.get("NVIDIA_VISIBLE_DEVICES") or "").strip()
    # "all"/"none"/"void" are the CDI keywords, not card indices
    if not devices or devices.lower() in ("all", "none", "void"):
        return host
    return f"{host}:{devices}"


def _build_run_tags(cfg) -> list:
    """Filterable W&B tags describing what actually varies between runs.

    Beyond the architecture/dataset/seed identity, this records the loss
    composition and the knobs that distinguish the current sweep, because a
    dozen runs whose names differ only by a suffix are unreadable in the UI
    otherwise. The commit is tagged too: an earlier round had ablations
    launched from checkouts at different commits, and that was only caught by
    checksumming prediction files afterwards.

    Every tag is "key:value", so the W&B tag list sorts by key and a filter
    can match a whole axis ("seed:", "machine:worker08"). Runs started before
    2026-10-02 carry the older bare tags (s123, fold1, worker07:0).
    """
    arch = getattr(cfg, "architecture", None)
    model_name = str(getattr(arch, "model_name", "unknown"))
    family = _model_family(model_name)
    tags = [f"arch:{model_name}"]
    if model_name != family:
        tags.append(f"family:{family}")

    use_kbrs = _is_kbrs_enabled(cfg)
    tags.append(f"kbrs:{'on' if use_kbrs else 'off'}")
    tags.append(f"win:{cfg.window_size}")
    dataset = str(_get_choice("dataset") or "?")
    tags.append(f"fold:{dataset[4:]}" if dataset.startswith("fold") else f"dataset:{dataset}")
    tags.append(f"seed:{cfg.seed}")
    tags.append(f"batch:{getattr(cfg, 'batch_size', '?')}")
    tags.append(f"sched:{getattr(cfg, 'lr_schedule', '?')}")
    mode_targets = _mode_targets(cfg)
    if mode_targets != "none":
        tags.append(f"targets:modes_{mode_targets}")

    # Which of the optional objectives are actually on. This is the ablation
    # axis, and reading it off the run name is error-prone.
    weights = getattr(arch, "loss_weights", None) or {}
    active = [k for k in ("rmc", "rep", "sm") if float(weights.get(f"lambda_{k}", 0.0)) > 0.0]
    if "director" in model_name.lower():
        tags.append("loss:" + ("+".join(active) if active else "hcm_only"))
        lam_sm = float(weights.get("lambda_sm", 0.0))
        if lam_sm > 0.0:
            tags.append(f"lambda_sm:{lam_sm:g}")

    # Non-default knobs only, so the tag list stays short when nothing is swept.
    for key, default, fmt in (
        ("dense_positives", False, lambda v: "pos:dense"),
        ("head_conv", config.DIRECTOR_HEAD_CONV, lambda v: f"head_conv:{v}"),
        ("centernet_down_ratio", 4, lambda v: f"stride:{v}"),
        ("render_sigma", config.DIRECTOR_RENDER_SIGMA, lambda v: f"sigma:{v:g}"),
        ("trainable_layers", config.DIRECTOR_TRAINABLE_LAYERS, lambda v: f"trainable_layers:{v}"),
        ("conf_threshold", config.DIRECTOR_TAU, lambda v: f"tau:{v:g}"),
        ("hcm_negative_target", config.DIRECTOR_HCM_NEGATIVE_TARGET, lambda v: f"hcm_neg:{v}"),
    ):
        value = getattr(arch, key, default)
        if value != default:
            tags.append(fmt(value))

    if use_kbrs:
        for group, key in (("plugins/kbrs/loss", "kbrs_loss"), ("plugins/kbrs/score", "kbrs_score"),
                           ("kbrs_loss", "kbrs_loss"), ("kbrs_score", "kbrs_score")):
            choice = _get_choice(group)
            if choice and not any(t.startswith(f"{key}:") for t in tags):
                tags.append(f"{key}:{str(choice).replace('/', '_')}")
        # The knobs that tell apart KBRS variants trained from the same group
        # choices; a run differing only here was otherwise indistinguishable.
        kp = getattr(getattr(cfg, "kbrs", None), "kbrs_params", None) or {}
        tags.append(f"kbrs_gate:{'on' if list(kp.get('gate_channels') or []) else 'off'}")
        tags.append(f"kbrs_mix:{kp.get('mixture_nonneg') or 'raw'}")

    commit = _git_state().get("commit")
    if commit:
        tags.append(f"git:{commit[:9]}")

    host = _host_tag()
    if host:
        tags.append(f"machine:{host}")

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
        "host": _host_tag(),
        "loss_weights": {k: float(v) for k, v in weights.items()} if weights else None,
    }
    for knob in (
        "down_ratio", "k_max", "conf_threshold", "render_sigma", "u_observers",
        "smooth_huber_delta", "smooth_warmup_start", "smooth_warmup_full",
        "soft_center_radius", "peak_border_margin", "trainable_layers",
        "head_conv", "dense_positives", "hcm_negative_target",
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


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

CHECKPOINT_FORMAT = 2


def save_checkpoint(path, model, optimizer, lr_scheduler, epoch) -> None:
    """Write a checkpoint that a run can actually be continued from.

    Weights alone are not enough to resume. SGD carries momentum buffers, and
    the cosine schedule's position is held in the scheduler, not derived from
    anything the weights record - restart without it and the rate jumps back to
    its initial value, which at two thirds through a run is four times what it
    should be. A resumed run would then be a different experiment from an
    uninterrupted one, which matters most for the seed sweeps, where the whole
    point is that the runs differ only by seed.

    `epoch` is the number of epochs completed, so training continues from it.
    """
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": int(epoch),
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, lr_scheduler=None, device=None) -> int:
    """Restore a checkpoint and return the epoch to continue from.

    Checkpoints written before this was a dict hold a bare state_dict. Those
    load their weights and return 0: there is no optimizer or schedule state to
    recover, so continuing from one is a fresh run that happens to start from
    trained weights, and it says so rather than pretending otherwise.
    """
    blob = torch.load(path, map_location=device or "cpu")

    if not isinstance(blob, dict) or "model" not in blob:
        model.load_state_dict(blob)
        Logger.warn(
            f"[Resume] {path} predates optimizer state; weights loaded but the "
            "optimizer and schedule restart. This is not a faithful "
            "continuation - do not mix it into a seed comparison."
        )
        return 0

    model.load_state_dict(blob["model"])
    if optimizer is not None and blob.get("optimizer") is not None:
        optimizer.load_state_dict(blob["optimizer"])
    if lr_scheduler is not None and blob.get("lr_scheduler") is not None:
        lr_scheduler.load_state_dict(blob["lr_scheduler"])

    epoch = int(blob.get("epoch", 0))
    Logger.info(f"[Resume] {path}: continuing from epoch {epoch}")
    return epoch


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _epoch_log(epoch: int, train_stats) -> dict:
    """The W&B scalars of one epoch: total loss, NaN counters, every loss_* term."""
    log_dict = {
        "epoch": epoch,
        "Loss/train": train_stats.loss.global_avg,
    }
    for k in ("skipped", "nan", "grad_nonfinite", "grad_norm"):
        m = getattr(train_stats, k, None)
        if m is not None and hasattr(m, "global_avg"):
            log_dict[f"Train/{k}"] = float(m.global_avg)
    for k, meter in getattr(train_stats, "meters", {}).items():
        if k.startswith("loss_") and hasattr(meter, "global_avg"):
            log_dict[f"Loss/{k[5:]}"] = float(meter.global_avg)  # loss_classifier -> Loss/classifier
    return log_dict


def train_model(
    model,
    optimizer,
    lr_scheduler,
    data_loader_train,
    device,
    num_epochs,
    save_dir,
    id_string: str = "",
    checkpoint_every: int = 10,
    start_epoch: int = 0,
):
    """Train from start_epoch to num_epochs, logging every epoch to W&B and
    saving a checkpoint every `checkpoint_every` epochs and at the last one."""
    _write_run_provenance(save_dir, model, id_string)

    Logger.info("[Stage] Starting training loop...")
    if start_epoch:
        Logger.info(f"[Stage] Resuming at epoch {start_epoch} of {num_epochs}")
    for epoch in tqdm.tqdm(range(start_epoch, num_epochs), initial=start_epoch,
                           total=num_epochs):
        # Loss terms that ramp over training (currently only Director's
        # L_smooth) read the epoch off the module. Walking .modules() rather
        # than calling model.set_epoch() keeps this working through
        # KBRSWrapper and any future wrapper without each one forwarding it.
        for _module in model.modules():
            if hasattr(_module, "set_epoch"):
                _module.set_epoch(epoch)

        t0 = time.time()
        train_stats = train_one_epoch(
            model,
            optimizer,
            data_loader_train,
            device,
            epoch,
            print_freq=PRINT_FREQ,
            scaler=None,
            nan_log_path=os.path.join(save_dir, "nan_batches.jsonl"),
            skip_nonfinite=True,
            max_consecutive_nan=MAX_CONSECUTIVE_NAN,
            grad_clip_norm=float(getattr(config, "GRAD_CLIP_NORM", 0.0)),
            lr_backoff=LR_BACKOFF_ON_NAN,
            retry_fp32_on_nan=True,
        )
        Logger.info(f"[Time][epoch {epoch}] train_one_epoch: {time.time() - t0:.1f}s")

        lr_scheduler.step()

        t_wandb = time.time()
        wandb.log(_epoch_log(epoch, train_stats), commit=True)
        Logger.info(f"[Time][epoch {epoch}] wandb.log (scalars): {time.time() - t_wandb:.3f}s")

        # Saving every epoch would leave 30 ResNet-50 copies per run. Only the
        # last one is read (inference finds it by --epoch), so intermediate
        # ones are kept every `checkpoint_every` epochs for tracing the
        # learning curve, and the last epoch is always saved.
        is_interval = checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0
        if is_interval or (epoch + 1) == num_epochs:
            tc0 = time.time()
            save_path = os.path.join(save_dir, f"model_{epoch+1:03d}.pth")
            save_checkpoint(save_path, model, optimizer, lr_scheduler, epoch + 1)
            Logger.info(f"[Info] Saved model checkpoint: {save_path} (time: {time.time() - tc0:.2f}s)")

        try:
            send_message(f"[{id_string}] Epoch {epoch+1} completed.")
        except Exception as e:
            Logger.error(f"Failed to send message: {e}")


# ---------------------------------------------------------------------------
# Setup steps of run_training, in the order they run
# ---------------------------------------------------------------------------

def _seed_everything(cfg) -> int:
    """Seed Python, NumPy and torch from cfg.seed (drawing one if unset)."""
    seed = cfg.seed
    if seed is None:
        seed = secrets.randbits(31)
        cfg.seed = seed
        Logger.info(f"[Seed] No seed provided in config; generated seed={seed}")
    else:
        Logger.info(f"[Seed] Using seed={seed}")

    # torchvision's CUDA roi_align kernel is non-deterministic, so with
    # torch.use_deterministic_algorithms enabled `ops.roi_align` silently takes a
    # pure-Python decomposition instead. That path materializes a
    # [K, C, PH, PW, IY, IX] intermediate: Mask R-CNN training asks for 5.5 GiB in
    # a single allocation and dies on a 32 GiB card, at batch 16 as well as 32.
    # inference.py already opts out for the same reason; training has to do the
    # same for the models that reach roi_align. The heatmap models never call it,
    # so they keep determinism, which is what makes their seed comparison mean
    # anything.
    arch_name = str(getattr(cfg.architecture, "model_name", "")).lower()
    uses_roi_align = "rcnn" in arch_name
    if uses_roi_align:
        Logger.warn(
            f"[Seed] deterministic algorithms disabled for architecture "
            f"'{arch_name}': they force torchvision's roi_align onto a "
            f"pure-Python fallback that OOMs. Runs of this architecture are "
            f"reproducible only up to the kernel's own non-determinism."
        )
    set_global_seed(int(seed), deterministic=not uses_roi_align)
    return int(seed)


def _select_device(cfg) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
    Logger.info(f"[Info] Using device: {device} (torch.cuda.is_available(): {torch.cuda.is_available()} / cfg.cuda: {cfg.cuda})")
    return device


def _start_run(cfg) -> str:
    """Name the run (cfg.id_string), create its directory and start W&B.
    Returns the directory checkpoints and provenance go to."""
    # Tags are built for every run, including those launched with an explicit
    # id_string (every ablation); they used to be built only when id_string was
    # empty, which left those runs unfilterable in W&B.
    run_tags = _build_run_tags(cfg)
    if not cfg.id_string:
        cfg.id_string = "_".join(run_tags + [time.strftime("%Y%m%d_%H%M%S")])
        Logger.info(f"[Info] Using id string: {cfg.id_string}")
    Logger.info(f"[Info] W&B tags: {run_tags}")

    save_dir = os.path.join(cfg.log_root, f"{cfg.id_string}/")
    os.makedirs(save_dir, exist_ok=True)
    Logger.info(f"[Info] Log save path: {save_dir}")

    if wandb.run is not None:
        wandb.finish()
    wandb.init(
        project="starcraft",
        name=cfg.id_string,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=run_tags,
    )
    return save_dir


@dataclass
class TrainingData:
    loader: DataLoader
    in_channels: int   # channels per frame x frames per window
    mode_targets: str  # cfg.mode_targets, validated


def _prepare_data(cfg, seed: int) -> TrainingData:
    """Build the label (and, if needed, mode) caches and the training loader.

    The ranked-mode cache is needed by Director-CenterNet (its targets) and by
    Mask R-CNN trained with mode_targets=hard/soft (its boxes).
    """
    input_root = os.path.join(cfg.data_root, "input/dst")
    label_root = os.path.join(cfg.data_root, "label/dst")
    train_replays = list(cfg.dataset.train_replays)
    test_replays = list(getattr(cfg.dataset, "test_replays", []) or [])
    all_replays = sorted(set(str(r) for r in (train_replays + test_replays)))

    ensure_label_pickles(
        label_root=label_root,
        label_method=cfg.label_method,
        replay_ids=all_replays,
        num_workers=cfg.num_workers,
    )
    Logger.info("[Info] JSON to Pickle conversion completed.")

    is_director = "director" in str(getattr(cfg.architecture, "model_name", "")).lower()
    mode_targets = _mode_targets(cfg)
    needs_modes = is_director or mode_targets != "none"
    if needs_modes:
        from dataset.mode_cache import ensure_mode_cache
        ensure_mode_cache(
            label_root=label_root,
            label_method=cfg.label_method,
            replay_ids=all_replays,
            sigma=float(getattr(cfg, "mode_extraction_sigma", config.MODE_EXTRACTION_SIGMA)),
            min_sep=float(getattr(cfg, "mode_extraction_min_sep", config.MODE_EXTRACTION_MIN_SEP)),
            rel_threshold=float(getattr(cfg, "mode_extraction_rel_threshold", config.MODE_EXTRACTION_REL_THRESHOLD)),
            max_modes=int(getattr(cfg, "mode_extraction_max_modes", config.MODE_EXTRACTION_MAX_MODES)),
            num_workers=cfg.num_workers,
        )

    # The validation loader (val_count windows of the test replays) is built
    # but not evaluated during training; it is kept because building it is
    # part of load_data's seeded split.
    data_loader_train, data_loader_validation, _ = load_data(
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
        seed=seed,
        pair_mode=is_director,  # consecutive windows, for L_smooth
        use_mode_cache=needs_modes,
        mode_targets=mode_targets,
    )
    n_val = len(data_loader_validation.dataset) if data_loader_validation is not None else "(none)"
    Logger.info(f"[Info] Data loaded: Train {len(data_loader_train.dataset)}, Validation {n_val}")

    dataset = data_loader_train.dataset
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    in_channels = len(dataset.channel_indices) * dataset.window_size
    Logger.info(f"[Info] Input channels: {in_channels} (window size: {dataset.window_size})")
    return TrainingData(data_loader_train, in_channels, mode_targets)


def _loss_weights(cfg) -> dict:
    """architecture.loss_weights, overridden by a top-level loss_weights.

    The KBRS weight (plugins.kbrs.loss_weights) is not read here; KBRSWrapper
    falls back to 0.25, the weight of every reported KBRS run.
    """
    loss_weights = {}
    for source in (getattr(cfg.architecture, "loss_weights", None),
                   cfg.loss_weights if "loss_weights" in cfg else None):
        if source is not None:
            for name, weight in source.items():
                loss_weights[name] = float(weight)
    return loss_weights


def _kbrs_params(cfg) -> Optional[dict]:
    """plugins.kbrs.kbrs_params, overridden by a top-level kbrs_params; None without KBRS."""
    if not _is_kbrs_enabled(cfg):
        return None
    kbrs_params = OmegaConf.to_container(cfg.kbrs.kbrs_params, resolve=True)
    if "kbrs_params" in cfg and cfg.kbrs_params is not None:
        extra = cfg.kbrs_params
        kbrs_params.update(OmegaConf.to_container(extra, resolve=True) if isinstance(extra, DictConfig) else dict(extra))
    return kbrs_params


# What build_model reads from the architecture config, when it is set there.
_ARCHITECTURE_KEYS = (
    "centernet_down_ratio", "max_objs",
    "k_max", "conf_threshold", "render_sigma", "u_observers",
    "smooth_huber_delta", "smooth_warmup_start", "smooth_warmup_full",
    "soft_center_radius", "peak_border_margin", "trainable_layers",
    "head_conv", "dense_positives", "hcm_negative_target",
)


def _model_args(cfg, in_channels: int, mode_targets: str) -> DictConfig:
    """Everything build_model reads, in one place.

    A key left out here takes build_model's default. Notably resize_mode is
    never set, so Mask R-CNN resizes its input to 640x640 even though
    conf/architecture/maskrcnn.yaml says "resize" (800); all v6 runs were
    trained that way. A DictConfig (not a dict or namespace) is returned
    because that is what the model received when it read these off the Hydra
    config directly: loss_weights and kbrs_params reach it as DictConfigs.
    """
    args = {
        "model_name": getattr(cfg.architecture, "model_name", "maskrcnn"),
        "use_kbrs": cfg.use_kbrs,
        "use_density_peak": getattr(cfg.architecture, "use_density_peak", getattr(cfg, "use_density_peak", False)),
        "window_size": cfg.window_size,
        "in_channels": in_channels,
        "num_classes": NUM_CLASSES,
        "kbrs_params": _kbrs_params(cfg),
        "loss_weights": _loss_weights(cfg),
        "soft_mode_cls": mode_targets == "soft",
    }
    for key in _ARCHITECTURE_KEYS:
        if hasattr(cfg.architecture, key):
            args[key] = getattr(cfg.architecture, key)
    return OmegaConf.create(args)


def _build_model(cfg, data: TrainingData, device) -> torch.nn.Module:
    Logger.info("[Stage] Initializing model...")
    model = build_model(_model_args(cfg, data.in_channels, data.mode_targets))
    Logger.info(f"[Info] Model initialized with {NUM_CLASSES} classes and {data.in_channels} input channels.")
    model.to(device)
    return model


def _build_optimizer_and_schedule(cfg, model):
    """SGD, and a cosine (default) or step LR schedule over max_epoch."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=cfg.learning_rate, momentum=SGD_MOMENTUM, weight_decay=SGD_WEIGHT_DECAY
    )
    # StepLR(step_size=3, gamma=0.1) stepped once per epoch, which over a
    # 30-epoch run drops the rate by a decade every three epochs: 5e-3 for
    # epochs 1-3, 5e-4 through 6, 5e-5 through 9, and 5e-12 by epoch 30. The
    # run was effectively over after epoch six. Three consequences were
    # visible in the results before the cause was:
    #
    #   - L_smooth ramps from epoch 5 and reaches full weight at epoch 10,
    #     where the rate is a thousandth of its initial value. The objective
    #     had never actually been trained at the weight it is reported with.
    #   - the same holds, less sharply, for every optional objective, which is
    #     why the ablation lattice was flat to within seed noise.
    #   - a randomly initialized decoder between the pretrained backbone and
    #     the heads cannot converge in six epochs, which is how the stride-2
    #     run collapsed to 66% frame coverage with a median peak score of
    #     0.275 against 0.572 at stride 4.
    #
    # Cosine over max_epoch keeps the rate useful across the whole run and has
    # no cliff for a warmup to land behind. engine_safe already applies the
    # torchvision linear warmup within epoch 0, so this only shapes the decay.
    schedule = str(getattr(cfg, "lr_schedule", "cosine")).lower()
    if schedule == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(cfg.max_epoch)
        )
    elif schedule == "step":
        # The legacy shape, rescaled so the three decade drops span the run
        # instead of its first sixth.
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, int(cfg.max_epoch) // 3), gamma=0.1
        )
    else:
        raise ValueError(f"Unknown lr_schedule '{schedule}'; use cosine or step.")
    Logger.info(
        f"[Info] LR schedule: {schedule}, base lr={cfg.learning_rate}, "
        f"max_epoch={cfg.max_epoch}"
    )
    return optimizer, lr_scheduler


def _maybe_resume(cfg, model, optimizer, lr_scheduler, device) -> int:
    """Restore cfg.resume if set; returns the epoch to start from (0 otherwise).

    The scheduler is restored rather than fast-forwarded, so the rate picks up
    exactly where it stopped; a fresh cosine restarted two thirds through a run
    would be four times too high. Momentum comes back with the optimizer state,
    so the continuation is faithful and a resumed run stays comparable to one
    that ran straight through.
    """
    resume_path = getattr(cfg, "resume", None)
    if not resume_path:
        return 0
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
    start_epoch = load_checkpoint(resume_path, model, optimizer, lr_scheduler, device)
    if start_epoch >= int(cfg.max_epoch):
        raise ValueError(
            f"{resume_path} is already at epoch {start_epoch}, which is "
            f"max_epoch ({cfg.max_epoch}); nothing left to run."
        )
    return start_epoch


def run_training(cfg: DictConfig):
    """Train one model as configured; see the module docstring for the steps."""
    settings.update({"wandb": True})
    Logger.info("[Stage] Preparing environment...")

    seed = _seed_everything(cfg)
    device = _select_device(cfg)
    save_dir = _start_run(cfg)
    data = _prepare_data(cfg, seed)
    model = _build_model(cfg, data, device)
    optimizer, lr_scheduler = _build_optimizer_and_schedule(cfg, model)
    start_epoch = _maybe_resume(cfg, model, optimizer, lr_scheduler, device)

    train_model(
        model,
        optimizer,
        lr_scheduler,
        data.loader,
        device,
        cfg.max_epoch,
        save_dir,
        id_string=cfg.id_string,
        checkpoint_every=int(getattr(cfg, "checkpoint_every", 10)),
        start_epoch=start_epoch,
    )
    torch.cuda.empty_cache()
    send_message(f"@work Training run '{cfg.id_string}' completed successfully.")
    wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def prepare_cfg(cfg: DictConfig) -> None:
    """Resolve the switches that select inputs, once, before anything reads cfg.

    ROCI lives in its own label file, so the toggle becomes a label method
    name here. Everything downstream - the dataset, the inference pass, the
    run's id_string - then follows without needing to know the flag exists,
    and a run is named after the targets it actually trained on.
    """
    if cfg.get("roci", False) and not str(cfg.label_method).endswith("_roci"):
        cfg.label_method = f"{cfg.label_method}_roci"


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    prepare_cfg(cfg)
    print(OmegaConf.to_yaml(cfg))  # the full resolved config, at the top of the log
    Logger.set_level(cfg.log_level)

    try:
        Logger.info("[Entry] Starting training script...")
        run_training(cfg)
    except Exception as e:
        # id_string may still be empty if the failure came before _start_run
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
