import argparse
import os
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import time

# Load all labels from the specified directory
def load_all_labels(path):
    label_path = os.path.join(path, path.split('/')[-2])
    label_files = sorted(os.listdir(label_path))
    labels = []
    
    for file in label_files:
        label_data = np.load(os.path.join(label_path, file), allow_pickle=True)[1]
        labels.extend(label_data)
    
    return np.array(labels)

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, path, transforms, window_size):
        self.window_size = window_size
        self.label_sequences, self.dir_paths = self._load_sequences(path)
        self.seq_indexs = self._create_seq_indices()

    def _load_sequences(self, path):
        label_sequences = []
        dir_paths = []

        for directory in os.listdir(path):
            full_dir_path = os.path.join(path, directory)
            if os.path.isdir(full_dir_path):
                label_sequences.append(load_all_labels(full_dir_path))
                dir_paths.append(full_dir_path)
                print(f"Loaded {directory}")
        
        return label_sequences, dir_paths

    def _create_seq_indices(self):
        seq_indexs = []
        start = 0

        for i, seq in enumerate(self.label_sequences):
            end = start + len(seq) - (self.window_size - 1) - 150
            seq_indexs.append((i, start, end))
            start = end
        
        return seq_indexs

    def __len__(self):
        return self.seq_indexs[-1][-1]

    def __getitem__(self, idx):
        for i, start, end in self.seq_indexs:
            if start <= idx < end:
                real_idx = idx - start + 150
                entire_data = np.load(os.path.join(self.dir_paths[i], f"{real_idx}.npy"), allow_pickle=True)
                data, masks = self._extract_masks(entire_data)
                break

        input_data = self.preprocessing(data)
        target = self._create_target(masks, idx)

        return input_data, target

    def _extract_masks(self, entire_data):
        masks = [i[0] for i in entire_data if i != 0]
        return entire_data[0][0], np.stack(masks)

    def preprocessing(self, data):
        temp = np.zeros([self.window_size, 9, data.shape[2], data.shape[2]])
        temp[:, 0] = data[0]
        temp[:, 1] = data[1]
        temp[:, 2] = data[2]
        temp[:, 3] = data[4]
        temp[:, 4] = data[6]
        temp[:, 5] = data[7]
        temp[:, 6] = data[8]
        temp[:, 7] = data[10]
        temp[:, 8] = data[13]

        data = temp.reshape(self.window_size * temp.shape[1], temp.shape[2], -1)
        return torch.FloatTensor(data)

    def _create_target(self, masks, idx):
        num_objs = 5
        boxes = self._get_boxes(masks, num_objs)

        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        return target

    def _get_boxes(self, masks, num_objs):
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        return torch.as_tensor(boxes, dtype=torch.float32)


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.backbone.body.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main(args):
    torch.cuda.empty_cache()

    data_path = args.load_dir
    log_save_path = os.path.join(
        args.log_save_dir,
        f"model_{args.id_string}_lr{args.learning_rate}_w_size{args.window_size}_{str(int(time.time()))[4:]}/"
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    dataset = PennFudanDataset(data_path, get_transform(train=False), window_size=1)
    dataset_test = PennFudanDataset(data_path, get_transform(train=False), window_size=1)

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    data_path_test = "./saved_models/"
    os.makedirs(data_path_test, exist_ok=True)

    for epoch in range(args.max_epoch):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)
        torch.save(model.state_dict(), os.path.join(data_path_test, f"model_{epoch}.pth"))

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object detection training with Mask R-CNN")
    parser.add_argument("--log-save-dir", type=str, default="./saved_models/")
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--load-dir", type=str, default="./trainig_data_five_several/")
    
    #
    parser.add_argument("--eval", action="store_true")
    
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0001)

    parser.add_argument("--id-string", type=str, default="")
    # 안씀
    parser.add_argument("--cuda-idx", type=int, default=0)
    args = parser.parse_args()

    main(args)
