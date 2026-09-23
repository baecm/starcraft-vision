from argparse import ArgumentParser

from utils import Viewport

import config

def parse_arguments():
    parser = ArgumentParser(description="Generate viewport-based point labels for replays")
    parser.add_argument("--replays", type=str, nargs="+", required=True, help="Replay ID(s) to process")
    parser.add_argument("--method", type=str, default=config.LABEL_METHODS[3], choices=config.LABEL_METHODS, help="Viewport label extraction method")
    roci = parser.add_mutually_exclusive_group()
    roci.add_argument("--roci", dest="roci", action="store_true",
                      help="add Joo et al.'s regions of common interest as extra "
                           "targets. Writes <method>_roci.json beside the plain "
                           "labels; a model must be trained from scratch on it, "
                           "since this changes the training targets.")
    roci.add_argument("--no-roci", dest="roci", action="store_false",
                      help="plain observer viewports only")
    parser.set_defaults(roci=config.ROCI_ENABLED)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    # Output the parsed arguments for verification
    print(f"[INFO] Method: {args.method} (roci={args.roci})")

    for replay_id in args.replays:
        with Viewport(replay_id) as viewport:
            print(f"[INFO] Processing {replay_id}")
            if not viewport.load():
                print(f"[SKIP] Replay {replay_id}: No viewport data")
                continue
            viewport.run(method=args.method, roci=args.roci)
            viewport.save()

if __name__ == "__main__":
    main()
