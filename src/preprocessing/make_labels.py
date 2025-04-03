from argparse import ArgumentParser
import data
from common import config


def process_replay(replay_id: int, src_dir: str, dst_dir: str, method: str):
    """단일 리플레이에 대해 viewport 라벨 처리"""
    vp_data = data.viewport.load(src_dir, replay_id)

    if len(vp_data) == 0:
        print(f"[SKIP] Replay {replay_id}: No viewport data")
        return

    vp_processed = data.viewport.preprocess(vp_data, method)
    data.viewport.store_npy(vp_processed, dst_dir, replay_id, method)
    print(f"[SAVE] Replay {replay_id}: Stored with method='{method}'")


def main(args):
    for replay_id in args.replays:
        process_replay(replay_id, args.src_dir, args.dst_dir, args.method)


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate viewport-based point labels for replays")
    parser.add_argument("--replays", type=int, nargs="+", required=True, help="Replay ID(s) to process")
    parser.add_argument("--method", type=str, default=config.label_method[0],
                        choices=config.label_method, help="Viewport label extraction method")
    parser.add_argument("--src-dir", type=str, default="../data/src/vpds/", help="Source directory for vpds files")
    parser.add_argument("--dst-dir", type=str, default="../data/dst/", help="Destination directory for label npy")

    args = parser.parse_args()
    main(args)
