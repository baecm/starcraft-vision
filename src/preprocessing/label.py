from argparse import ArgumentParser

from utils import Viewport


def parse_arguments():
    parser = ArgumentParser(description="Generate viewport-based point labels for replays")
    parser.add_argument("--replays", type=str, nargs="+", required=True, help="Replay ID(s) to process")
    parser.add_argument("--method", type=str, default=config.LABEL_METHODS[0], choices=config.LABEL_METHODS, help="Viewport label extraction method")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    # Output the parsed arguments for verification
    print(f"[INFO] Method: {args.method}")
    
    for replay_id in args.replays:
        with Viewport(replay_id) as viewport:
            print(f"[INFO] Processing {replay_id}")
            if not viewport.load():
                print(f"[SKIP] Replay {replay_id}: No viewport data")
                continue
            viewport.run(method=args.method)
            viewport.save()

if __name__ == "__main__":
    main()
