# src/config.py
import sys
from enum import Enum

GRAD_CLIP_NORM = 2.0

LABEL_METHODS = [
    'legacy',
    'consider_previous',
    'unique_local_maximums',
    'all_correct',
]

OUTPUT_TYPES = ['coord', 'channel']

KERNEL_SHAPE = (20, 12)
ORIGIN_SHAPE = (128, 128)
TILE_SIZE = 32


class Channel(Enum):
    Player_1_Worker = 0
    Player_1_Ground = 1
    Player_1_Air = 2
    Player_1_Building = 3
    Player_2_Worker = 4
    Player_2_Ground = 5
    Player_2_Air = 6
    Player_2_Building = 7
    Resource = 8
    Vision = 9
    Terrain = 10


# Mapping from component names to channel indices
COMPONENT_CHANNEL_MAP = {
    'worker': [Channel.Player_1_Worker.value, Channel.Player_2_Worker.value],
    'ground': [Channel.Player_1_Ground.value, Channel.Player_2_Ground.value],
    'air': [Channel.Player_1_Air.value, Channel.Player_2_Air.value],
    'building': [Channel.Player_1_Building.value, Channel.Player_2_Building.value],
    'resource': [Channel.Resource.value],
    'vision': [Channel.Vision.value],
    'terrain': [Channel.Terrain.value],
}

PARAM_ALIAS_MAP = {
    "arch=": "architecture=",
    "arch.": "architecture.",
    "win=": "window_size=",
    "win_size=": "window_size=",
    "lr=": "learning_rate=",
    "batch=": "batch_size=",
    "ds=": "dataset=",
    "kbrs=": "plugins/kbrs=",
    "density_peak=": "plugins/density_peak=",
    "probabilistic_query=": "plugins/probabilistic_query=",
    "loss=": "plugins/kbrs/loss=",
    "kbrs_loss=": "plugins/kbrs/loss=",
    "score=": "plugins/kbrs/score=",
    "kbrs_score=": "plugins/kbrs/score=",
    "epoch=": "max_epoch=",
    "epochs=": "max_epoch=",
}


def resolve_cli_aliases(argv: list[str] = None) -> list[str]:
    """
    Preprocess command-line arguments to resolve short parameter key aliases.
    e.g., 'arch=centernet' -> 'architecture=centernet'
          'win=4' -> 'window_size=4'
          '--arch' -> '--model-name'
    """
    if argv is None:
        argv = sys.argv[1:]

    processed = []
    for arg in argv:
        new_arg = arg

        # 1. Hydra key=value overrides
        for short_k, full_k in PARAM_ALIAS_MAP.items():
            if new_arg.startswith(short_k):
                new_arg = full_k + new_arg[len(short_k):]
                break

        # 2. CLI dash flags
        if new_arg.startswith("--arch") and not new_arg.startswith("--architecture") and not new_arg.startswith("--arch-"):
            new_arg = new_arg.replace("--arch", "--model-name", 1)
        elif new_arg.startswith("--win") and not new_arg.startswith("--window-size"):
            new_arg = new_arg.replace("--win", "--window-size", 1)
        elif new_arg.startswith("--ds") and not new_arg.startswith("--dataset"):
            new_arg = new_arg.replace("--ds", "--replays", 1)

        processed.append(new_arg)

    return processed