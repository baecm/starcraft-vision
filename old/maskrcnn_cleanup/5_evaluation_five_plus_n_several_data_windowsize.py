import numpy as np
import numpy
import torch.nn as nn

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

import os
import numpy as np
import torch
from PIL import Image
import torch.nn as nn

import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# from torchsummary import summary

from engine import train_one_epoch, evaluate
import utils
import transforms as T

# import matplotlib.pyplot as plt
# import matplotlib

import numpy as np
import pandas as pd
from tqdm import tqdm

# matplotlib.use('TkAgg')

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.backbone.body.conv1 = nn.Conv2d(36, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


class PennFudanDataset(object):
    def __init__(self, path, window_size):
        pth = os.listdir(path)
        self.label_sequences = []

        self.dir_paths = []
        if os.path.isdir(path):
            self.dir_paths.append(path + '/')
            print(f"Loaded {path}")

        self.window_size = window_size
        self.seq_indexs = []
        self.seq_indexs.append((0, 0, len(pth)))

    def __len__(self):
        return self.seq_indexs[-1][-1]

    def __getitem__(self, idx):
        # load images and masks
        for i, start, end in self.seq_indexs:
            if idx >= start and idx < end:
                real_idx = idx - start

                if real_idx > 3:
                    entire_data1 = np.load(self.dir_paths[i] + '/' + str(int(real_idx) - 3) + ".npy", allow_pickle=True)
                    entire_data2 = np.load(self.dir_paths[i] + '/' + str(int(real_idx) - 2) + ".npy", allow_pickle=True)
                    entire_data3 = np.load(self.dir_paths[i] + '/' + str(int(real_idx) - 1) + ".npy", allow_pickle=True)
                    entire_data4 = np.load(self.dir_paths[i] + '/' + str(int(real_idx) + 0) + ".npy", allow_pickle=True)
                else:
                    entire_data1 = np.load(self.dir_paths[i] + '/' + str(int(real_idx)) + ".npy", allow_pickle=True)
                    entire_data2 = entire_data1
                    entire_data3 = entire_data1
                    entire_data4 = entire_data1

                data1 = entire_data1[0]
                data2 = entire_data2[0]
                data3 = entire_data3[0]
                data4 = entire_data4[0]

                # masks = entire_data4[1]
                break

        input_data = self.preprocessing(data1, data2, data3, data4)
        return input_data

    def preprocessing(self, data1, data2, data3, data4):
        # 0 ground 1 air 2 building 3 spell 4 ground 5 air 6 building 7 spell 8 resource 9 vision 10 terrain
        temp = np.zeros([self.window_size, 9, data1.shape[1], data1.shape[2]])

        temp[0] = data1
        temp[1] = data2
        temp[2] = data3
        temp[3] = data4

        data = temp
        data = data.reshape(self.window_size * data.shape[1], data.shape[2], -1)
        # #data = data.reshape(self.window_size*data.shape[0],data.shape[1],-1)
        # label = np.array([label[0]/3456, label[1]/3720])
        return torch.FloatTensor(data)


import sys
# data_path = [f"../result/6254"]
#data_path = [f"./trainig_data2/36/",f"./trainig_data2/212/",f"./trainig_data2/438/",f"./trainig_data2/522/",f"./trainig_data2/1660/"]

# for i in data_path:
#     print(i.split('/')[-1])

def main(args):
    # parser.add_argument("--training", type=int, nargs="+")
    # parser.add_argument("--load-dir", type=str, default=f"../result/")
    # data_path = [f"../result/6254"]
    for i in args.testing:
        i = "../result/" + str(i)
        print(i)
        # pth = os.listdir(i)
        replay_name = int(i.split('/')[-1])
        dataset = PennFudanDataset(i, get_transform(train=False), window_size=4)

        dataset_len = len(dataset)
        #     dataset_len = 21458
        Start, End, Step = 0, len(dataset), 1
        test_img_array = []
        test_img_one_channel_array = []
        test_target_array = []

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        num_classes = 2
        model = get_model_instance_segmentation(num_classes)
        model.load_state_dict(torch.load(args.saved_model, map_location=torch.device(device)))
        model.to(device)
        model.eval()

        print("dataset_size:", End)
        for i in range(Start, End, Step):
            img_t = dataset[i]
            test_img_array.append(img_t)
            img_one_channel = img_t.sum(axis=0, keepdim=True)
            test_img_one_channel_array.append(img_one_channel)
        #         test_target_array.append(target_t)

        print("input load")
        vpx_array = []
        vpy_array = []

        with torch.no_grad():
            for idx, i in tqdm(enumerate(test_img_array)):
                prediction = model(torch.unsqueeze(i, 0).to(device))
                if prediction[0]["boxes"].shape[0] == 0:
                    prediction = model(torch.unsqueeze(i - 1, 0).to(device))
                    vpx = int(prediction[0]["boxes"][0][0]) * 32
                    vpy = int(prediction[0]["boxes"][0][1]) * 32
                else:
                    vpx = int(prediction[0]["boxes"][0][0]) * 32
                    vpy = int(prediction[0]["boxes"][0][1]) * 32
                if idx % 500 == 0:
                    print("idx:  ", idx)

                vpx_array.append(vpx)
                vpy_array.append(vpy)

        #     dataset_len = 500

        save_vpd_dir = args.save_dir
        os.makedirs(save_vpd_dir, exist_ok=True)
        temp = np.zeros((dataset_len, 1))
        for i in range(0, dataset_len):
            temp[i] = int(i * 8)
        temp2 = np.zeros((int(temp.max()), 1))
        for i in range(0, int(temp.max())):
            temp2[i] = i

        dataset_temp = pd.DataFrame({"frame": temp[:, 0], "vpx": vpx_array[:], "vpy": vpy_array[:]})
        dataset_temp2 = pd.DataFrame({"frame": temp2[:, 0]})
        dataset = pd.merge(left=dataset_temp2, right=dataset_temp, how="left", on="frame")
        dataset = dataset.fillna(method="ffill")
        dataset.to_csv(save_vpd_dir + str(replay_name) + ".rep.vpd", header=True, index=False)
        #     dataset.to_csv("C:/TM/starcraft/bwapi-data/read/preset/" + str(replay_name) + "_0413.rep.vpd", header= True, index=False)

        print("saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--testing", type=int, nargs="+")
    parser.add_argument("--load-dir", type=str, default=f"../result/")
    parser.add_argument("--save-dir", type=str, default=f"./saved_xy/five_plus_n_sung/")
    parser.add_argument("--saved-model", type=str, default=f"./saved_models/five_plus_n_sung/model_20.pth")
    args = parser.parse_args()
    main(args)