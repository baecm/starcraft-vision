import math
import sys
import time

import torch
import utils

from detection.engine import _get_iou_types

from .coco_utils import get_coco_api_from_dataset
from .custom_evaluator import CustomEvaluator

from utils.logger import Logger
    

@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    Logger.info("[Eval] Creating COCO API from dataset...")
    coco = get_coco_api_from_dataset(data_loader.dataset)

    iou_types = _get_iou_types(model)
    custom_evaluator = CustomEvaluator(coco, iou_types)

    Logger.info("[Eval] Start inference on validation set...")
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, outputs)
        }

        for t in targets:
            Logger.debug(f"[Eval] image_id: {t['image_id']} → {t['image_id'].item()}")

        evaluator_time = time.time()
        custom_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

        Logger.debug(f"[Eval] Processed batch with {len(images)} images")
        Logger.debug(f"[Eval] Model time: {model_time:.4f}s, Evaluator time: {evaluator_time:.4f}s")

        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    Logger.info("[Eval] Finished inference, synchronizing...")
    metric_logger.synchronize_between_processes()

    Logger.info("[Eval] Accumulating results...")
    custom_evaluator.synchronize_between_processes()
    custom_evaluator.accumulate()

    Logger.info("[Eval] Summarizing results...")
    custom_evaluator.summarize()

    torch.set_num_threads(n_threads)
    return custom_evaluator
