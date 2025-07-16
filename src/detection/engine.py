import math
import sys
import time

import torch
from torch.amp import autocast
import torchvision.models.detection.mask_rcnn
import utils

from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
from . import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    import time
    from utils.logger import Logger

    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    Logger.info("[Eval] Creating COCO API from dataset...")
    coco = get_coco_api_from_dataset(data_loader.dataset)

    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    Logger.info("[Eval] Start inference on validation set...")
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # --- image_id .item() 적용 필수 ---
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, outputs)
        }

        # 디버깅용 로그 (선택)
        for t in targets:
            Logger.debug(f"[Eval] image_id: {t['image_id']} → {t['image_id'].item()}")

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

        Logger.debug(f"[Eval] Processed batch with {len(images)} images")
        Logger.debug(f"[Eval] Model time: {model_time:.4f}s, Evaluator time: {evaluator_time:.4f}s")

        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    Logger.info("[Eval] Finished inference, synchronizing...")
    metric_logger.synchronize_between_processes()

    Logger.info("[Eval] Accumulating results...")
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()

    Logger.info("[Eval] Summarizing results...")
    coco_evaluator.summarize()

    torch.set_num_threads(n_threads)
    return coco_evaluator