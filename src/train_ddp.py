import os
import argparse
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import tqdm
import utils
from detection.utils import MetricLogger
import json
import wandb

from detection.engine import evaluate
from dataset.custom_penn_fudan import CustomPennFudanDataset
from model.maskrcnn_builder import get_model_instance_segmentation
import detection.transforms as T
from utils.logger import Logger
import config

from torch.cuda.amp import autocast, GradScaler

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def make_loader(ds, batch_size, sampler):
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=4,
        sampler=sampler,
        pin_memory=True,
        collate_fn=utils.collate_fn
    )

def load_data(input_root, label_root, label_method,
              window_size, batch_size,
              replay_ids=None, train_replays=None,
              test_replays=None, test_size=50,
              sample_ratio=1.0):
    Logger.info("[Stage] Loading data...")
    if replay_ids:
        all_ids = [str(r) for r in replay_ids]
    else:
        all_ids = [d[:-4] for d in os.listdir(input_root) if d.endswith('.rep')]

    if train_replays is not None:
        train_ids = [str(r) for r in train_replays]
        test_ids = [str(r) for r in test_replays] if test_replays else sorted(set(all_ids) - set(train_ids))
    else:
        train_ids = all_ids
        test_ids = all_ids

    train_dataset = CustomPennFudanDataset(input_root, label_root, label_method, training_ids=train_ids, window_size=window_size)
    test_dataset = CustomPennFudanDataset(input_root, label_root, label_method, training_ids=test_ids, window_size=window_size)

    if sample_ratio < 1.0:
        train_idx = torch.randperm(len(train_dataset)).tolist()[:int(len(train_dataset) * sample_ratio)]
        test_idx = torch.randperm(len(test_dataset)).tolist()[:int(len(test_dataset) * sample_ratio)]
        train_dataset = CustomPennFudanDataset(input_root, label_root, label_method, training_ids=train_ids, window_size=window_size, indices=train_idx)
        test_dataset = CustomPennFudanDataset(input_root, label_root, label_method, training_ids=test_ids, window_size=window_size, indices=test_idx)

    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = make_loader(train_dataset, batch_size, train_sampler)
    test_loader = make_loader(test_dataset, 1, test_sampler)
    return train_loader, test_loader, train_sampler

def train_model(model, optimizer, lr_scheduler,
                data_loader_train, data_loader_test,
                device, num_epochs, save_dir, train_sampler, is_main):
    scaler = GradScaler()
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        model.train()
        metric_logger = MetricLogger(delimiter="  ")
        header = f"Epoch: [{epoch}]"

        for images, targets in metric_logger.log_every(data_loader_train, 10, header):
            images = list(img.to(device) for img in images)
            targets = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]
            
            with autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            metric_logger.update(loss=losses, **loss_dict)

        lr_scheduler.step()
        eval_stats = evaluate(model, data_loader_test, device=device)

        if is_main:
            log_dict = {}
            for k, meter in metric_logger.meters.items():
                log_dict[f"Loss/{k}"] = meter.global_avg
            if isinstance(eval_stats, dict):
                for k, v in eval_stats.items():
                    log_dict[f"Eval/{k}"] = v
            wandb.log(log_dict)
            torch.save(model.module.state_dict(), os.path.join(save_dir, f"model_{epoch}.pth"))

def run_training(args):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

    if not args.id_string:
        args.id_string = f"{args.label_method}_win{args.window_size}_b{args.batch_size}"

    log_save_path = os.path.join(args.log_root, f"{args.id_string}_{time.strftime('%Y%m%d_%H%M%S')}/")
    if is_main:
        os.makedirs(log_save_path, exist_ok=True)
        wandb.init(project="starcraft", name=args.id_string, config=args)

    input_root = os.path.join(args.data_root, "input/dst")
    label_root = os.path.join(args.data_root, "label/dst")

    data_loader_train, data_loader_test, train_sampler = load_data(
        input_root, label_root, args.label_method,
        args.window_size, args.batch_size,
        replay_ids=args.replays,
        train_replays=args.train_replays,
        test_replays=args.test_replays,
        sample_ratio=args.sample_ratio
    )

    num_classes = 2
    model = get_model_instance_segmentation(num_classes, window_size=args.window_size)
    model.to(device)
    model = DDP(model, device_ids=[local_rank])

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_model(model, optimizer, lr_scheduler, data_loader_train, data_loader_test,
                device, args.max_epoch, log_save_path, train_sampler, is_main)
    
    if is_main:
        wandb.finish()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", type=str, nargs="+", required=True)
    parser.add_argument("--train-replays", type=str, nargs="+", default=None)
    parser.add_argument("--test-replays", type=str, nargs="+", default=None)
    parser.add_argument("--label-method", type=str, default=config.LABEL_METHODS[0], choices=config.LABEL_METHODS)
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_gument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--id-string", type=str, default="")
    parser.add_argument("--data-root", type=str, default=os.path.join(os.getcwd(), "data"))
    parser.add_argument("--log-root", type=str, default=os.path.join(os.getcwd(), "models"))
    parser.add_argument("--log-level", type=str, default="log", choices=["none", "log", "debug"])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    Logger.set_level(args.log_level)
    run_training(args)
