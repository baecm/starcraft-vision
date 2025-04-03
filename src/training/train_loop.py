import os
import torch
from models.maskrcnn_builder import get_model_instance_segmentation
from dataset.penn_fudan import PennFudanDataset
from transforms.base_transforms import get_transform
import utils

from detection.engine import *


def run_training(args):
    device = torch.device(f"cuda:{args.cuda_idx}" if args.cuda and torch.cuda.is_available() else "cpu")

    dataset = PennFudanDataset(
        args.load_dir, get_transform(is_train=True), args.window_size, args.training, mode=args.mode
    )
    dataset_val = PennFudanDataset(
        args.load_dir, get_transform(is_train=False), args.window_size, args.training, mode=args.mode
    )

    indices = torch.randperm(len(dataset)).tolist()
    split = 100 if len(indices) > 200 else int(len(indices) * 0.2)
    dataset = torch.utils.data.Subset(dataset, indices[:-split])
    dataset_val = torch.utils.data.Subset(dataset_val, indices[-split:])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn
    )

    model = get_model_instance_segmentation(num_classes=args.num_classes, in_channels=args.window_size * 9)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    os.makedirs(args.log_save_dir, exist_ok=True)

    for epoch in range(args.max_epoch):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        scheduler.step()
        evaluate(model, data_loader_val, device=device)

        if epoch % 2 == 0:
            save_path = os.path.join(args.log_save_dir, f"model_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Saved checkpoint to {save_path}")
