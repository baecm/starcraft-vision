"""The training loop: train_one_epoch_safe, with NaN/Inf handling. utils.py has its logging helpers."""
import json
import math
import os
import time
from typing import Any, Dict, List, Optional

import torch

from . import utils

try:
    from utils.logger import Logger
except ModuleNotFoundError:
    from src.utils.logger import Logger


def _to_py(x: Any) -> Any:
    """Convert to something JSON can serialize, as far as possible."""
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            try:
                return float(x.detach().cpu().item())
            except Exception:
                return str(x.detach().cpu())
        return {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
        }
    if isinstance(x, (list, tuple)):
        return [_to_py(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_py(v) for k, v in x.items()}
    return str(x)


def _append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _loss_dict_to_float(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in loss_dict.items():
        try:
            out[k] = float(v.detach().cpu().item())
        except Exception:
            out[k] = float("nan")
    return out


def _any_nonfinite_tensor(t: torch.Tensor) -> bool:
    return bool((~torch.isfinite(t)).any().item())


def _has_nonfinite_grads(params: List[torch.nn.Parameter]) -> bool:
    for p in params:
        if p.grad is None:
            continue
        if _any_nonfinite_tensor(p.grad):
            return True
    return False


def train_one_epoch_safe(
    model,
    optimizer,
    data_loader,
    device,
    epoch: int,
    progress_interval_s: float,
    scaler: Optional[torch.amp.GradScaler] = None,
    *,
    on_progress=None,
    nan_log_path: str = "",
    skip_nonfinite: bool = True,
    max_consecutive_nan: int = 20,
    grad_clip_norm: float = 0.0,
    lr_backoff: float = 0.5,
    min_lr: float = 1e-6,
    retry_fp32_on_nan: bool = True,
):
    """One training epoch that keeps going through NaN/Inf where it can.

    - A non-finite loss skips the batch (logged). With AMP (a GradScaler
      passed as `scaler`) it is first retried once in fp32; train.py passes
      no scaler, so its runs are fp32 throughout and never retry.
    - Non-finite gradients after backward skip the optimizer step (logged).
    - Too many consecutive skips abort the epoch instead of looping forever.
    - Optional gradient clipping, and an LR backoff after a non-finite batch.
    Skipped batches are recorded in nan_log_path (JSON lines) and printed.

    A progress line is printed every progress_interval_s seconds;
    on_progress(metric_logger, i, seconds_per_iter) is called with each one.
    """

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("skipped", utils.SmoothedValue(window_size=1, fmt="{value:.0f}"))
    metric_logger.add_meter("nan", utils.SmoothedValue(window_size=1, fmt="{value:.0f}"))
    metric_logger.add_meter("grad_nonfinite", utils.SmoothedValue(window_size=1, fmt="{value:.0f}"))
    metric_logger.add_meter("grad_norm", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = f"[Epoch {epoch + 1}]"  # 1-based, as in train.py's epoch summary

    # Seed the meters so the first print (count=0) does not divide by zero.
    metric_logger.update(
        lr=float(optimizer.param_groups[0]["lr"]),
        skipped=0.0,
        nan=0.0,
        grad_nonfinite=0.0,
        grad_norm=0.0,
    )

    # linear LR warmup over epoch 0, as in the torchvision reference engine
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    params = [p for p in model.parameters() if p.requires_grad]
    consecutive_nan = 0
    skipped_total = 0
    nan_total = 0
    grad_nf_total = 0

    def _batch_meta(targets: List[Dict[str, Any]]) -> Dict[str, Any]:
        # image_id at least; any tracing keys the dataset adds are recorded too
        out = {"image_id": []}
        for t in targets:
            iid = t.get("image_id")
            if isinstance(iid, torch.Tensor) and iid.numel() == 1:
                out["image_id"].append(int(iid.detach().cpu().item()))
            else:
                out["image_id"].append(_to_py(iid))
        # the usual tracing keys (rid, window, idx, ...), when present
        for k in ("rid", "window", "window_image_ids", "sample_idx", "idx"):
            vals = []
            ok = False
            for t in targets:
                if k in t:
                    ok = True
                    vals.append(_to_py(t.get(k)))
            if ok:
                out[k] = vals
        # number of target boxes
        out["num_boxes"] = []
        for t in targets:
            b = t.get("boxes")
            if isinstance(b, torch.Tensor):
                out["num_boxes"].append(int(b.shape[0]))
            else:
                out["num_boxes"].append(None)
        return out

    def _backoff_lr() -> Dict[str, float]:
        # lower the LR to damp a blow-up
        new_lrs = {}
        for i, g in enumerate(optimizer.param_groups):
            old = float(g.get("lr", 0.0))
            new = max(min_lr, old * lr_backoff) if lr_backoff > 0 else old
            g["lr"] = new
            new_lrs[f"group{i}"] = new
        return new_lrs

    def _report_skip(step: int, reason: str) -> None:
        # Printed as it happens; the batch details go to nan_log_path.
        Logger.warn(
            f"[NaN] epoch {epoch + 1} iter {step + 1}/{len(data_loader)}: {reason}; batch skipped, "
            f"lr backed off to {optimizer.param_groups[0]['lr']:.6g} "
            f"(skipped this epoch: {skipped_total})"
        )

    batches = metric_logger.log_every(
        data_loader, progress_interval_s, header,
        on_report=None if on_progress is None else (lambda i, s_per_it: on_progress(metric_logger, i, s_per_it)),
    )
    for step, (images, targets) in enumerate(batches):
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]

        # 1) forward (AMP optional)
        with torch.amp.autocast(device_type="cuda", enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduced across processes, for logging
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = float(losses_reduced.detach().cpu().item())

        # 2) loss non-finite handling
        if not math.isfinite(loss_value):
            nan_total += 1
            consecutive_nan += 1

            # retry in fp32 (helps mostly when AMP is on)
            if retry_fp32_on_nan and scaler is not None:
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    loss_dict2 = model(images, targets)
                    losses2 = sum(loss for loss in loss_dict2.values())
                loss_dict2r = utils.reduce_dict(loss_dict2)
                losses2r = sum(loss for loss in loss_dict2r.values())
                loss2 = float(losses2r.detach().cpu().item())
                if math.isfinite(loss2):
                    # finite in fp32: continue with that loss
                    loss_dict = loss_dict2
                    losses = losses2
                    loss_dict_reduced = loss_dict2r
                    losses_reduced = losses2r
                    loss_value = loss2
                    consecutive_nan = 0
                else:
                    # the retry failed too: skip the batch
                    pass

            if not math.isfinite(loss_value):
                rec = {
                    "ts": time.time(),
                    "kind": "loss_nonfinite",
                    "epoch": int(epoch),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "loss": loss_value,
                    "loss_dict": _loss_dict_to_float(loss_dict_reduced),
                    "meta": _batch_meta(targets),
                }
                # cheap input statistics
                try:
                    rec["input_max_abs"] = [
                        float(img.detach().abs().amax().cpu().item()) for img in images
                    ]
                except Exception:
                    pass
                _append_jsonl(nan_log_path, rec)

                # LR backoff
                new_lrs = _backoff_lr()
                _append_jsonl(nan_log_path, {"ts": time.time(), "kind": "lr_backoff", "epoch": int(epoch), "new_lrs": new_lrs})

                skipped_total += 1
                _report_skip(step, f"non-finite loss ({loss_value})")
                metric_logger.update(skipped=float(skipped_total), nan=float(nan_total))
                metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])

                optimizer.zero_grad(set_to_none=True)

                if not skip_nonfinite or consecutive_nan >= max_consecutive_nan:
                    raise RuntimeError(
                        f"Non-finite loss encountered (epoch={epoch}) and cannot continue: loss={loss_value}, consecutive={consecutive_nan}"
                    )
                continue

        # the loss is finite from here on
        consecutive_nan = 0

        # 3) backward + grad checks
        optimizer.zero_grad(set_to_none=True)
        grad_norm = None
        if scaler is not None:
            scaler.scale(losses).backward()
            # unscale first, so the gradients can be clipped and checked
            try:
                scaler.unscale_(optimizer)
            except Exception:
                pass
            if grad_clip_norm and grad_clip_norm > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, grad_clip_norm).detach().cpu().item())
            if _has_nonfinite_grads(params):
                grad_nf_total += 1
                rec = {
                    "ts": time.time(),
                    "kind": "grad_nonfinite",
                    "epoch": int(epoch),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "loss": float(losses_reduced.detach().cpu().item()),
                    "loss_dict": _loss_dict_to_float(loss_dict_reduced),
                    "meta": _batch_meta(targets),
                }
                _append_jsonl(nan_log_path, rec)
                _backoff_lr()
                skipped_total += 1
                _report_skip(step, "non-finite gradients")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()  # adjust the scale
                metric_logger.update(skipped=float(skipped_total), grad_nonfinite=float(grad_nf_total))
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, grad_clip_norm).detach().cpu().item())
            if _has_nonfinite_grads(params):
                grad_nf_total += 1
                rec = {
                    "ts": time.time(),
                    "kind": "grad_nonfinite",
                    "epoch": int(epoch),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "loss": float(losses_reduced.detach().cpu().item()),
                    "loss_dict": _loss_dict_to_float(loss_dict_reduced),
                    "meta": _batch_meta(targets),
                }
                _append_jsonl(nan_log_path, rec)
                _backoff_lr()
                skipped_total += 1
                _report_skip(step, "non-finite gradients")
                optimizer.zero_grad(set_to_none=True)
                metric_logger.update(skipped=float(skipped_total), grad_nonfinite=float(grad_nf_total))
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                continue
            optimizer.step()

        # 4) warmup step
        if lr_scheduler is not None:
            lr_scheduler.step()

        # 5) meters
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(skipped=float(skipped_total), nan=float(nan_total), grad_nonfinite=float(grad_nf_total))
        if grad_norm is not None and math.isfinite(grad_norm):
            metric_logger.update(grad_norm=grad_norm)

    return metric_logger
