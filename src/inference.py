#!/usr/bin/env python
"""
Run a trained checkpoint over replays and write COCO-style prediction files.

    make inference ARGS="--model-name <run> --model-number 30 --replays 275 1725 ..."

Writes <output-dir>/<model-name>/model_<NNN>[_th<score-threshold>]/<replay>.rep/<label-method>.json,
plus seed.txt and inference_provenance.json next to them. evaluate.py and the
analysis scripts read that layout (evaluate.load_coco_preds).

run_inference does, in order:
  1. seed and device
  2. checkpoint path, output directory, provenance     _checkpoint_path, _prepare_run_dir
  3. what to build: input channels, architecture, tau  _input_channels, _resolve_architecture,
                                                       _resolve_decoder_threshold
  4. build the model and load the weights              _load_model
  5. per replay: predict and save                      _predict_replay, save_predictions_as_coco
"""

import gc
import json
import multiprocessing
import os
import secrets
from types import SimpleNamespace

import torch
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

import config
from cli import parse_inference_args
from dataset.inference_dataset import InferenceDataset
from models.factory import build_model
from utils.logger import Logger
from utils.provenance import write_provenance
from utils.seed import set_global_seed
from utils.synology_chat import send_message
from utils.torch_compat import disable_inductor

# Before any model runs: torchvision's roi_align compile hook otherwise takes
# Mask R-CNN inference down on images without a C compiler.
disable_inductor()

NUM_CLASSES = 2  # background + viewport


def collate_fn(batch):
    """
    Bundle a batch into the format ([images], [metadata]).
    The metadata is a list of (replay_id, frame_id) tuples.
    """
    images, metas = zip(*batch)
    return list(images), list(metas)


def _available_cpu_count() -> int:
    """Estimate usable CPU cores (affinity-aware if possible)."""
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return multiprocessing.cpu_count()


def _auto_num_workers(device: torch.device) -> int:
    """
    Recommended DataLoader workers:
    - GPU: max(1, avail-1)  to keep I/O pipeline busy without oversubscription
    - CPU: max(0, avail-1)  to avoid contention with compute
    """
    avail = _available_cpu_count()
    if device.type == "cuda":
        return max(1, avail - 1)
    return max(0, avail - 1)


# ---------------------------------------------------------------------------
# Building the model a checkpoint was trained as
# ---------------------------------------------------------------------------

# Keys that train.py records in run_provenance.json, mapped to the attribute
# names build_model reads. Only knobs that change what the model *is* or how it
# decodes belong here; training-only settings (render_sigma, dense_positives,
# the L_smooth shaping) are deliberately absent.
_PROVENANCE_ARCH_KEYS = {
    "head_conv": "head_conv",
    "down_ratio": "centernet_down_ratio",
    "peak_border_margin": "peak_border_margin",
}


def _apply_recorded_architecture(model_args: SimpleNamespace, model_folder: str) -> None:
    """Rebuild the model the way the checkpoint was trained, in place.

    build_model takes a flat namespace rather than the Hydra config, because
    inference runs outside Hydra. The consequence is that every architecture
    knob has to be forwarded by hand: a run trained with
    `architecture.head_conv=256` would otherwise be rebuilt at the
    src/config.py default of 64 and load_state_dict would fail on a shape
    mismatch, after the training had already finished.

    train.py writes those knobs next to the weights, so read them from there
    instead of widening the CLI every time a knob is added. Missing file means
    a checkpoint from before provenance existed; the config defaults then apply
    as they did before.
    """
    path = os.path.join(model_folder, "run_provenance.json")
    if not os.path.isfile(path):
        Logger.warn(
            f"[Inference] No run_provenance.json in {model_folder}; "
            "rebuilding from src/config.py defaults. If this checkpoint was "
            "trained with a non-default architecture knob, the load will fail."
        )
        return

    try:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
    except Exception as e:
        Logger.warn(f"[Inference] Could not read {path}: {e}")
        return

    applied = {}
    for key, attr in _PROVENANCE_ARCH_KEYS.items():
        if key in record:
            setattr(model_args, attr, record[key])
            applied[attr] = record[key]
    if applied:
        Logger.info(f"[Inference] Architecture from run_provenance.json: {applied}")

    commit = (record.get("git") or {}).get("commit")
    if commit:
        Logger.info(f"[Inference] Checkpoint was trained at commit {commit[:9]}")


def _architecture_from_state_dict(state) -> "str | None":
    """Identify the architecture from the saved parameter names.

    The caller's guess comes from substring-matching the run's id_string, so a
    name containing neither "director" nor "centernet" silently becomes
    "maskrcnn" - and load_state_dict(strict=False) then accepts the mismatch,
    leaving a freshly initialized Mask R-CNN that crashes later in roi_heads.
    The weights themselves are unambiguous, so ask them instead.
    """
    keys = list(state.keys())

    def has(prefix: str) -> bool:
        return any(k.startswith(prefix) for k in keys)

    if has("roi_heads.") or has("rpn."):
        return "maskrcnn"
    if has("hm_head."):
        # Director sits on a ResNet-FPN; the plain CenterNet backbone is a bare
        # nn.Sequential and so has no fpn submodule.
        return "director_centernet" if has("backbone.fpn.") else "centernet"
    return None


def _load_model(model_path: str, device: torch.device, in_channels: int, window_size: int,
                architecture: str, num_classes: int = NUM_CLASSES, use_kbrs: bool = False,
                kbrs_params: dict = None, k_max: int = 3, conf_threshold: float = 0.2):
    """Build the model a checkpoint was trained as and load its weights (eval mode)."""
    # Settle the architecture before building anything: building the wrong one
    # and loading non-strict fails much later and much less obviously.
    state = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    elif 'model' in state:
        state = state['model']

    # A run trained with the KBRS plugin saves the KBRSWrapper, whose detector
    # parameters sit under "base_model.". KBRS only adds a training loss, so the
    # detector alone is what predicts; unwrap it unless the wrapper is built too.
    if not use_kbrs and any(k.startswith("base_model.") for k in state):
        detector = {k[len("base_model."):]: v for k, v in state.items()
                    if k.startswith("base_model.")}
        dropped = len(state) - len(detector)
        Logger.info(f"[Inference] KBRS checkpoint: loading the wrapped detector "
                    f"({len(detector)} tensors, {dropped} wrapper-only dropped)")
        state = detector

    sniffed = _architecture_from_state_dict(state)
    if sniffed and sniffed != architecture.lower():
        Logger.warn(
            f"[Inference] Checkpoint parameters say '{sniffed}' but the caller asked for "
            f"'{architecture}'; using '{sniffed}'."
        )
        architecture = sniffed
    elif sniffed:
        Logger.info(f"[Inference] Architecture confirmed from checkpoint: {sniffed}")

    # What build_model reads (models/factory.py). Anything left out takes its
    # default there, as in training: notably Mask R-CNN's 640x640 input.
    model_args = SimpleNamespace(
        model_name=architecture.lower(),
        use_kbrs=use_kbrs,
        num_classes=num_classes,
        in_channels=in_channels,
        window_size=window_size,
        kbrs_params=kbrs_params,
        loss_weights=None,
        k_max=k_max,
        conf_threshold=conf_threshold,
    )
    _apply_recorded_architecture(model_args, os.path.dirname(model_path))

    model = build_model(model_args)
    missing, unexpected = model.load_state_dict(state, strict=False)

    # strict=False is kept so a plugin wrapper can be attached or dropped, but
    # a mismatch of this size means the wrong architecture was built and the
    # run would otherwise proceed on mostly random weights.
    if len(missing) > 0:
        Logger.warn(f"[Inference] Missing keys: {len(missing)} items")
    if len(unexpected) > 0:
        Logger.warn(f"[Inference] Unexpected keys: {len(unexpected)} items")
    if len(missing) > 0 and len(unexpected) > 0:
        raise RuntimeError(
            f"[Inference] Checkpoint does not match the model that was built: "
            f"{len(missing)} missing and {len(unexpected)} unexpected keys. "
            f"Built '{architecture}' from {model_path}. Refusing to run inference on "
            f"partially initialized weights."
        )

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Writing predictions
# ---------------------------------------------------------------------------

def save_predictions_as_coco(
    replay_id: str,
    replay_results: list,
    label_method: str,
    output_dir: str,
    score_threshold: float = None,
):
    """
    Save a single replay's predictions in COCO format (masks omitted for compactness).
    """
    out_dir = os.path.join(output_dir, f"{replay_id}.rep")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label_method}.json")

    categories = [{"id": 1, "name": "viewport", "supercategory": "viewport"}]

    info_dict = {
        "description": f"Predictions for replay {replay_id}",
        "version": "1.0",
        "label_method": label_method,
    }
    if score_threshold is not None:
        info_dict["score_threshold"] = float(score_threshold)

    coco = {
        "info": info_dict,
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    # prevent duplicate image entries
    seen_frames = set()
    ann_id = 1

    for item in replay_results:
        fid = int(item["frame_id"])

        # images: single per frame
        if fid not in seen_frames:
            coco["images"].append({
                "id": fid,
                "file_name": f"{replay_id}.rep/{fid}.npy",
                "width": int(config.ORIGIN_SHAPE[1]),
                "height": int(config.ORIGIN_SHAPE[0]),
            })
            seen_frames.add(fid)

        # annotations (bbox-only; polygon is derived from bbox for viewer compatibility)
        for box, score, label in zip(item["boxes"], item["scores"], item["labels"]):
            x1, y1, x2, y2 = map(int, box)
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            segmentation = [[x1, y1, x1 + w, y1, x1 + w, y1 + h, x1, y1 + h]]

            coco["annotations"].append({
                "id": ann_id,
                "image_id": fid,
                "category_id": int(label),   # single class -> 1 also OK
                "bbox": [x1, y1, w, h],
                "score": float(score),
                "area": int(w * h),
                "segmentation": segmentation,
                "iscrowd": 0,
            })
            ann_id += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    Logger.info(f"[Inference] Saved predictions for replay {replay_id} -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Steps of run_inference, in the order they run
# ---------------------------------------------------------------------------

def _checkpoint_path(args) -> str:
    path = os.path.join(args.model_root, args.model_name, f"model_{int(args.model_number):03d}.pth")
    Logger.info(f"[Inference] Checkpoint path: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _prepare_run_dir(args, model_path: str) -> str:
    """Create the output directory and write seed.txt and inference_provenance.json.

    The directory is <output-dir>/<model-name>/model_<NNN>[_th<score-threshold>],
    unless --run-name replaces the part under --output-dir.
    """
    if not getattr(args, "output_dir", None):
        args.output_dir = "predictions"
    model_number = int(args.model_number)
    th_suffix = f"_th{args.score_threshold}" if getattr(args, "score_threshold", None) is not None else ""
    run_name = getattr(args, "run_name", None) or os.path.join(args.model_name, f"model_{model_number:03d}{th_suffix}")
    run_dir = os.path.join(args.output_dir, run_name)

    Logger.info(f"[Inference] Output run dir: {run_dir}")
    try:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "seed.txt"), "w", encoding="utf-8") as f:
            f.write(str(args.seed) + "\n")
    except Exception as e:
        Logger.warn(f"[Inference] Failed to write seed.txt: {e}")

    # A prediction file could not previously say what produced it, which is a
    # problem the moment a sweep is split over machines: a workstation that has
    # not pulled produces silently different predictions, and the only way to
    # notice was to checksum files against an older run and already suspect it.
    # Named apart from the model folder's own run_provenance.json, which
    # describes the training run rather than this pass over it.
    try:
        written = write_provenance(
            os.path.join(run_dir, "inference_provenance.json"),
            {
                "model_name": args.model_name,
                "model_number": model_number,
                "checkpoint": model_path,
                "label_method": args.label_method,
                "window_size": args.window_size,
                "score_threshold": getattr(args, "score_threshold", None),
                "conf_threshold": getattr(args, "conf_threshold", None),
                "k_max": getattr(args, "k_max", None),
                "sample_ratio": getattr(args, "sample_ratio", None),
                "replays": list(args.replays),
                "seed": args.seed,
            },
        )
        git = written.get("git", {})
        Logger.info(
            f"[Provenance] commit {str(git.get('commit'))[:9]}"
            f"{' (DIRTY)' if git.get('dirty') else ''}"
            f" on {written.get('host') or 'unknown host'}"
        )
        if git.get("dirty"):
            Logger.warn(
                "[Provenance] working tree is dirty; these predictions are not "
                "reproducible from a commit."
            )
    except Exception as e:
        Logger.warn(f"[Inference] Failed to write inference_provenance.json: {e}")
    return run_dir


def _input_channels(args) -> int:
    """Channels per window, read off a dataset of the first replay."""
    dataset = InferenceDataset(
        os.path.join(args.data_root, "input", "dst"),
        [args.replays[0]],
        window_size=args.window_size,
        include_components=args.include_components
    )
    return len(dataset.channel_indices) * dataset.window_size


def _kbrs_params(args):
    """KBRS parameters for --use-kbrs, or None without it.

    KBRS only adds a training loss, so it never changes predictions; this only
    decides how the checkpoint's wrapper is rebuilt. conf/model/kbrs.yaml no
    longer exists, so in practice the parameters are the two set here and the
    wrapper's defaults fill the rest.
    """
    if not getattr(args, "use_kbrs", False):
        return None
    try:
        yaml_path = os.path.join(os.path.dirname(__file__), "../conf/model/kbrs.yaml")
        if os.path.exists(yaml_path):
            kbrs_cfg = OmegaConf.load(yaml_path)
            if hasattr(kbrs_cfg, "kbrs_params"):
                kbrs_params = OmegaConf.to_container(kbrs_cfg.kbrs_params, resolve=True)
            else:
                kbrs_params = OmegaConf.to_container(kbrs_cfg, resolve=True)
        else:
            kbrs_params = {}
        kbrs_params["window_size"] = args.window_size
        kbrs_params.setdefault("per_window", 9)
    except Exception as e:
        Logger.warn(f"[Inference] Failed to load the KBRS YAML: {e}")
        kbrs_params = {}
    return kbrs_params


def _resolve_architecture(args) -> str:
    """--architecture if given, otherwise a guess from the run name.
    Either way _load_model checks it against the checkpoint's parameters."""
    arch_name = getattr(args, "architecture", None)
    if arch_name:
        Logger.info(f"[Inference] Architecture from --architecture: {arch_name}")
        return arch_name
    name = args.model_name.lower()
    if "director" in name:
        arch_name = "director_centernet"
    elif "centernet" in name:
        arch_name = "centernet"
    else:
        arch_name = "maskrcnn"
    Logger.warn(
        f"[Inference] No --architecture given; guessed '{arch_name}' from the run name "
        f"'{args.model_name}'. This is only reliable when the name contains the model's "
        f"own name."
    )
    return arch_name


def _resolve_decoder_threshold(args) -> float:
    """Director-CenterNet's peak threshold tau: --conf-threshold, else the score filter.

    tau defaults to the score filter, because a filter set above tau removes
    every region the model was willing to emit - which is how the auxiliary
    regions were silently unreachable at tau 0.2 against a filter of 0.3.
    """
    th = getattr(args, "score_threshold", None)
    if th is None:
        th = config.DIRECTOR_TAU
    tau = getattr(args, "conf_threshold", None)
    if tau is None:
        tau = float(th)
    elif float(tau) < float(th):
        Logger.warn(
            f"[Inference] conf_threshold={tau} is below score_threshold={th}: peaks in "
            f"[{tau}, {th}) are emitted by the model and then discarded by the filter. "
            f"Minority regions are the ones in that band."
        )
    return float(tau)


def _predict_replay(model, args, replay_id: str, device) -> list:
    """One entry per window of the replay: frame_id, and the boxes, scores and
    labels that clear --score-threshold (all of them if it is unset)."""
    dataset = InferenceDataset(
        os.path.join(args.data_root, "input", "dst"),
        [replay_id],
        window_size=args.window_size,
        include_components=args.include_components
    )
    if 0.0 < args.sample_ratio < 1.0:
        total_len = len(dataset)
        sample_size = int(total_len * args.sample_ratio)
        indices = torch.randperm(total_len).tolist()[:sample_size]
        dataset = Subset(dataset, indices)
        Logger.info(f"[Inference] Applied sampling: {sample_size}/{total_len} frames for replay {replay_id}")

    if args.workers is not None and args.workers >= 0:
        num_workers = args.workers
    else:
        num_workers = _auto_num_workers(device)
    Logger.info(f"[Inference] Dataset frames for {replay_id}: {len(dataset)}; num_workers={num_workers}")
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    results = []
    with torch.inference_mode():
        for images, metas in tqdm.tqdm(data_loader, desc="Running inference for replay", unit="batch",
                                       disable=None):  # terminal only; the log gets one line per replay
            images = [img.to(device, non_blocking=True) for img in images]
            outputs = model(images)
            for output, (_rid, frame_id) in zip(outputs, metas):
                scores = output["scores"].detach().cpu().numpy().tolist()
                boxes = output["boxes"].detach().cpu().numpy().tolist()
                labels = output["labels"].detach().cpu().numpy().tolist()
                if args.score_threshold is not None:
                    keep = [i for i, s in enumerate(scores) if s >= args.score_threshold]
                else:
                    keep = list(range(len(scores)))
                results.append({
                    "frame_id": frame_id,
                    "boxes": [boxes[i] for i in keep],
                    "scores": [scores[i] for i in keep],
                    "labels": [labels[i] for i in keep],
                })
            del outputs, images
    return results


def run_inference(args):
    """Predict every replay in args.replays with one checkpoint; see the module docstring."""
    Logger.info("[Inference] Starting...")

    if getattr(args, "seed", None) is None:
        args.seed = secrets.randbits(31)
        Logger.info(f"[Seed] No --seed provided for inference; generated seed={args.seed}")
    else:
        Logger.info(f"[Seed] Using provided inference seed={args.seed}")
    # deterministic=False on purpose. torchvision's CUDA roi_align kernel is
    # non-deterministic, so with torch.use_deterministic_algorithms enabled it
    # falls back to a pure-Python reference implementation that materializes a
    # [K, C, PH, PW, IY, IX] tensor - 2.3 GiB per call at batch 16, which OOMs
    # a 32 GiB card, and is orders of magnitude slower besides. That path is
    # written to be torch.compile'd, which needs a C compiler the image lacks.
    #
    # Nothing is lost here: inference runs fixed weights with no dropout and no
    # sampling, so the only non-determinism is float accumulation order inside
    # roi_align, which does not move any metric we report. Weight loading, data
    # order and any sampling stay seeded.
    set_global_seed(int(args.seed), deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    Logger.info(f"[Inference] Using device: {device}")

    model_path = _checkpoint_path(args)
    run_dir = _prepare_run_dir(args, model_path)
    in_channels = _input_channels(args)
    kbrs_params = _kbrs_params(args)
    architecture = _resolve_architecture(args)
    tau = _resolve_decoder_threshold(args)

    model = _load_model(
        model_path=model_path,
        device=device,
        in_channels=in_channels,
        window_size=args.window_size,
        architecture=architecture,
        num_classes=NUM_CLASSES,
        use_kbrs=getattr(args, "use_kbrs", False),
        kbrs_params=kbrs_params,
        k_max=getattr(args, "k_max", None) or config.DIRECTOR_K,
        conf_threshold=tau,
    )

    for replay_id in args.replays:
        Logger.info(f"--- Processing replay: {replay_id} ---")
        results = _predict_replay(model, args, replay_id, device)
        save_predictions_as_coco(
            replay_id=replay_id,
            replay_results=results,
            label_method=args.label_method,
            output_dir=run_dir,
            score_threshold=args.score_threshold,
        )
        del results
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            send_message(f"@work [Inference] Completed {replay_id}. Predictions saved at {run_dir}")
        except Exception as e:
            Logger.error(f"[Inference] Error sending message: {e}")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    Logger.info("[Inference] Complete!")


def main():
    args = parse_inference_args()
    run_inference(args)


if __name__ == "__main__":
    main()
