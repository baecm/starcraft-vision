import os
import numpy as np
from argparse import ArgumentParser

import data
from common import config
from common import channel


def choose_data_channels(channels: np.ndarray):
    dch = channel.DataChannel

    return np.stack([
        channels[:, dch.PLAYER_1_UNITS_WORKER],
        # channels[:, dch.PLAYER_1_UNITS_TRIVIAL],
        channels[:, dch.PLAYER_1_UNITS_GROUND],
        channels[:, dch.PLAYER_1_UNITS_AIR],
        channels[:, dch.PLAYER_1_BUILDINGS],
        # channels[:, dch.PLAYER_1_SPELLS],
        channels[:, dch.PLAYER_2_UNITS_WORKER],
        # channels[:, dch.PLAYER_2_UNITS_TRIVIAL],
        channels[:, dch.PLAYER_2_UNITS_GROUND],
        channels[:, dch.PLAYER_2_UNITS_AIR],
        channels[:, dch.PLAYER_2_BUILDINGS],
        # channels[:, dch.PLAYER_2_SPELLS],

        # channels[:, dch.PLAYER_1_UNITS_WORKER_SPREAD],
        # channels[:, dch.PLAYER_1_UNITS_TRIVIAL_SPREAD],
        # channels[:, dch.PLAYER_1_UNITS_GROUND_SPREAD],
        # channels[:, dch.PLAYER_1_UNITS_AIR_SPREAD],
        # channels[:, dch.PLAYER_1_BUILDINGS_SPREAD],
        # channels[:, dch.PLAYER_1_SPELLS_SPREAD],
        # channels[:, dch.PLAYER_2_UNITS_WORKER_SPREAD],
        # channels[:, dch.PLAYER_2_UNITS_TRIVIAL_SPREAD],
        # channels[:, dch.PLAYER_2_UNITS_GROUND_SPREAD],
        # channels[:, dch.PLAYER_2_UNITS_AIR_SPREAD],
        # channels[:, dch.PLAYER_2_BUILDINGS_SPREAD],
        # channels[:, dch.PLAYER_2_SPELLS_SPREAD],

        # channels[:, dch.NEUTRAL_RESOURCE_FIELDS],         # 24 # raw data end
        channels[:, dch.NEUTRAL_VISION],                    # 25
        # channels[:, dch.NEUTRAL_TERRAIN]                  # 26
    ], axis=1)


def merge_result(terrain: np.ndarray, raw: np.ndarray, vision: np.ndarray):
    length = raw.shape[0]
    terrain = np.concatenate([terrain.reshape((1, 1,) + terrain.shape)] * length)
    return np.concatenate([raw, vision, terrain], axis=1)


def store_npz(channels: np.ndarray, dst_path: str, replay: int):
    os.makedirs(dst_path, exist_ok=True)
    np.savez_compressed(os.path.join(dst_path, str(replay)) + ".rep.channels_compressed", data=channels)


def main(args):
    for replay in args.replays:
        src_path = os.path.join(args.src_dir, str(replay))
        dst_path = os.path.join(args.dst_dir, str(replay))

        terrain = data.terrain.load(src_path, replay)
        raw = data.raw.load(src_path, replay)#.head(1000)
        vision = data.vision.load(src_path, replay)#.head(3)

        terminal_frame = data.raw.get_terminal_frame(raw)

        terrain = data.terrain.preprocess(terrain)
        raw = data.raw.preprocess(raw, args.interval, terrain.shape)
        vision = data.vision.preprocess(vision, args.interval, terrain.shape, terminal_frame)

        channels = merge_result(terrain, raw, vision)
        channels = choose_data_channels(channels)

        store_npz(channels, dst_path, replay)
        ## can add another store options by function ##

#
# python 1_make_input.py --replays 212
# python 2_make_label.py --replays 3613 --method all_correct
# python 3_make_paired.py --replays 3613  --method all_correct
#

#

#

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--replays", type=int, nargs="+", required=True)
    parser.add_argument("--src-dir", type=str, default="../data/src/")
    parser.add_argument("--dst-dir", type=str, default="../data/dst/")
    # parser.add_argument("--dst-dir", type=str, default="../data2/dst2/")
    # parser.add_argument("--dst-dir", type=str, default="/media/joo/Data/new/")
    parser.add_argument("--interval", type=int, default=config.INTERVAL)
    args = parser.parse_args()
    main(args)
