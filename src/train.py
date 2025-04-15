import os
import time
import torch
import tqdm
import utils

from detection.engine import train_one_epoch, evaluate
# from dataset import PennFudanDataset
from dataset.custom_penn_fudan import CustomPennFudanDataset

import detection.transforms as T


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def load_data(replays_dir, window_size, batch_size, all_replays=None, train_replays=None, test_replays=None, test_size=50):
    if train_replays is not None:
        if test_replays is None:
            test_replays = list(sorted(set(all_replays) - set(train_replays)))
            print(f"[Auto] Using replays {test_replays} for testing")

        train_dataset = CustomPennFudanDataset(replays_dir, training=train_replays, window_size=window_size)
        test_dataset = CustomPennFudanDataset(replays_dir, training=test_replays, window_size=window_size)

    else:
        print("[Auto] No train-replays provided, falling back to random split")
        full_dataset = CustomPennFudanDataset(replays_dir, training=os.listdir(replays_dir), window_size=window_size)
        indices = torch.randperm(len(full_dataset)).tolist()
        train_indices, test_indices = indices[:-test_size], indices[-test_size:]
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    def make_loader(ds, batch_size, shuffle):
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=utils.collate_fn)

    return make_loader(train_dataset, batch_size, shuffle=True), make_loader(test_dataset, batch_size=1, shuffle=False)


def train_model(model, optimizer, lr_scheduler, data_loader_train, data_loader_test, device, num_epochs, save_dir):
    for epoch in tqdm.tqdm(range(num_epochs)):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        
        lr_scheduler.step()
        
        evaluate(model, data_loader_test, device=device)
        
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_{epoch}.pth"))
        
    print("Training complete!")


def run_training(args):
    # Set device to CUDA if available and specified by the user
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Set up logging and save path
    log_save_path = os.path.join(
        args.log_root_dir,
        f"model_{args.id_string}_lr{args.learning_rate}_w_size{args.window_size}_{str(int(time.time()))[4:]}/"
    )
    # Load data
    data_loader_train, data_loader_test = load_data(args.replays, args.window_size, args.batch_size)

    # Define model
    num_classes = 2
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # Set up optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Create directory to save models
    os.makedirs(log_save_path, exist_ok=True)

    # Train the model
    train_model(model, optimizer, lr_scheduler, data_loader_train, data_loader_test, device, args.max_epoch, log_save_path)
