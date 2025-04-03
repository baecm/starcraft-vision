import os
import numpy as np
from argparse import ArgumentParser

import data
from common import config
from common import channel


def select_channels(channels: np.ndarray) -> np.ndarray:
    """원하는 채널만 선택하여 stack 형태로 반환"""
    ch = channel.DataChannel
    return np.stack([
        channels[:, ch.PLAYER_1_UNITS_WORKER],
        channels[:, ch.PLAYER_1_UNITS_GROUND],
        channels[:, ch.PLAYER_1_UNITS_AIR],
        channels[:, ch.PLAYER_1_BUILDINGS],
        channels[:, ch.PLAYER_2_UNITS_WORKER],
        channels[:, ch.PLAYER_2_UNITS_GROUND],
        channels[:, ch.PLAYER_2_UNITS_AIR],
        channels[:, ch.PLAYER_2_BUILDINGS],
        channels[:, ch.NEUTRAL_VISION],
    ], axis=1)


def merge_channels(terrain: np.ndarray, raw: np.ndarray, vision: np.ndarray) -> np.ndarray:
    """raw, vision, terrain을 하나의 배열로 합침"""
    n_frames = raw.shape[0]
    terrain_seq = np.repeat(terrain.reshape(1, 1, *terrain.shape), n_frames, axis=0)
    return np.concatenate([raw, vision, terrain_seq], axis=1)


def save_compressed(channels: np.ndarray, dst_dir: str, replay_id: int):
    os.makedirs(dst_dir, exist_ok=True)
    dst_file = os.path.join(dst_dir, f"{replay_id}.rep.channels_compressed.npz")
    np.savez_compressed(dst_file, data=channels)


def process_replay(replay_id: int, src_dir: str, dst_dir: str, interval: int):
    src_path = os.path.join(src_dir, str(replay_id))
    dst_path = os.path.join(dst_dir, str(replay_id))

    terrain = data.terrain.preprocess(data.terrain.load(src_path, replay_id))
    raw = data.raw.load(src_path, replay_id)
    vision = data.vision.load(src_path, replay_id)

    terminal_frame = data.raw.get_terminal_frame(raw)
    raw = data.raw.preprocess(raw, interval, terrain.shape)
    vision = data.vision.preprocess(vision, interval, terrain.shape, terminal_frame)

    channels = merge_channels(terrain, raw, vision)
    selected = select_channels(channels)

    save_compressed(selected, dst_path, replay_id)


def main(args):
    for replay_id in args.replays:
        process_replay(replay_id, args.src_dir, args.dst_dir, args.interval)


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract and save replay channel features")
    parser.add_argument("--replays", type=int, nargs="+", required=True, help="Replay ID(s) to process")
    parser.add_argument("--src-dir", type=str, default="../data/src/", help="Source directory")
    parser.add_argument("--dst-dir", type=str, default="../data/dst/", help="Destination directory")
    parser.add_argument("--interval", type=int, default=config.INTERVAL, help="Sampling interval")
    args = parser.parse_args()

    main(args)
