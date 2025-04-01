import os
import numpy as np

from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser

from common import config
from data import viewport


def scatter_labels_on_channel(labels_over_frame: list, kernel_shape=config.KERNEL_SHAPE, origin_shape=config.ORIGIN_SHAPE):
    kernel = np.ones(kernel_shape)

    result = []
    for labels in labels_over_frame:
        channels = []
        for label in labels:
            channel = np.zeros(origin_shape)
            channel[label[0]:label[0] + kernel_shape[0], label[1]:label[1] + kernel_shape[1]] += kernel
            channels.append(channel.T)
        result.append(channels)
    return result


def load_input_channels(data_dir: str, replay: int):
    input_path = os.path.join(data_dir, "{}/".format(replay))
    input_channels = np.load(input_path + "{}.rep.channels_compressed.npz".format(replay))['data']
    return list(input_channels)


def load_labels(data_dir: str, replay: int, method: str, output: str):
    label_path = os.path.join(data_dir, "{}/".format(replay), "{}/".format(method))
    labels = glob(label_path + "*.npy")
    result = [np.load(label_path + "{}.vpds.npy".format(i)) for i in range(len(labels))]
    if output == "coord":
        return result
    elif output == "channel":
        return scatter_labels_on_channel(result)
    return

def make_pairs(input_channels: list, labels: list):
    input_channels = list(input_channels)
    if len(labels) == 0:
        return zip(input_channels, [[None, None]] * len(input_channels)), len(input_channels)
    if len(input_channels) > len(labels):
        print("labels: ",len(labels))
        print("input_channels: ", len(input_channels))
        input_channels = input_channels[:len(labels)] #, :, :, :]
    else:
        labels = labels[:len(input_channels)]
    return zip(input_channels, labels), len(input_channels)

def store_pairs(pairs: zip, length:int,  result_path: str, replay: int, method: str):
    result_path = os.path.join(result_path, "{}/".format(replay))#, "{}/".format(method))
    os.makedirs(result_path, exist_ok=True)

    with tqdm(total=length, desc="Store pairs...") as pbar:
        for idx, pair in enumerate(pairs):
            pair = [pair[0], pair[1]]
            tmp = np.empty(len(pair), dtype=object)
            tmp[:] = pair
            np.save(result_path + "{}.npy".format(idx), arr=np.array(tmp, dtype=object))
            pbar.update(1)

def main(args):
    for replay in args.replays:
        input_channels = load_input_channels(args.data_dir, replay)
        labels = load_labels(args.data_dir, replay, args.method, args.output)
        pairs, length = make_pairs(input_channels, labels)
        store_pairs(pairs, length, args.result_dir, replay, args.method)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--replays", type=int, nargs="+")
    parser.add_argument("--method", type=str, default=config.label_method[0], choices=config.label_method)
    parser.add_argument("--output", type=str, default=config.output_method[0], choices=config.output_method)
    parser.add_argument("--data-dir", type=str, default="../data/dst/")
    # parser.add_argument("--dst-dir", type=str, default="/media/joo/Data/new/")
    #parser.add_argument("--result-dir", type=str, default="/media/joo/Data/result/")
    parser.add_argument("--result-dir", type=str, default="../result/")
    # parser.add_argument("--result-dir", type=str, default="../result/")
    args = parser.parse_args()
    main(args)
    # --replays 275 1725 3613 4520 4664 212 36 438 522 1660 --method all_correct
    # --replays 1559 1628 2351 6219 11251 --method all_correct