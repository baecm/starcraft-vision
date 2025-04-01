from argparse import ArgumentParser

import data
from common import config


def main(args):
    for replay in args.replays:
        viewport_data = data.viewport.load(args.src_dir, replay)
        # viewport_data = data.viewport.preprocess(viewport_data, args.method)
        if len(viewport_data) > 0:
            viewport_data = data.viewport.preprocess(viewport_data, args.method)
        data.viewport.store_npy(viewport_data, args.dst_dir, replay, args.method)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--replays", type=int, nargs="+", required=True)
    #parser.add_argument("--method", type=str, default="all_correct")
    parser.add_argument("--method", type=str, default=config.label_method[0], choices=config.label_method)
    parser.add_argument("--src-dir", type=str, default="../data/src/vpds/")
    # parser.add_argument("--dst-dir", type=str, default="/media/joo/Data/new/")
    parser.add_argument("--dst-dir", type=str, default="../data/dst/")
    args = parser.parse_args()

    main(args)
