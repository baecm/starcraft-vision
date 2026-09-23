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

# Director-CenterNet & Mode Extraction defaults
VIEWPORT_SIZE_HW = (12, 20)  # (height, width) in tiles
MODE_EXTRACTION_SIGMA = 4.0
MODE_EXTRACTION_MIN_SEP = 12.0
MODE_EXTRACTION_REL_THRESHOLD = 0.35
MODE_EXTRACTION_MAX_MODES = 5
NUM_OBSERVERS_U = 5

# ===== ROCI (Region of Common Interest) target augmentation =====
# Joo et al. (2023) add viewports where the human observers overlap as extra
# detection targets. Off by default: turning it on changes the labels, so the
# generated file is named "<method>_roci" and a model trained on it must be
# trained from scratch - an existing checkpoint cannot be converted.
ROCI_ENABLED = False
# The peak-finding parameters are the mode-extraction ones above, because the
# two procedures are the same. They are named separately so that ROCI's target
# generation and the evaluator's mode recovery can be moved apart later without
# one silently following the other.
ROCI_SIGMA = MODE_EXTRACTION_SIGMA
ROCI_MIN_SEP = MODE_EXTRACTION_MIN_SEP
# Joo et al. put their floor just above 1: a tile one observer watches scores
# 1, a tile two watch concurrently scores 2. This is read as a multiple of what
# one observer is worth under the same smoothing, so the 1.1 keeps that meaning
# regardless of sigma or viewport size and never needs recalibrating.
ROCI_THRESHOLD = 1.1
# Joo et al. cap the added regions at the number of observers.
ROCI_MAX_REGIONS = NUM_OBSERVERS_U

# Gaussian width for the rendered targets, in tiles. The heatmap grid samples
# every `down_ratio` tiles, so 2.0 tiles is half a cell: adjacent cells fall to
# exp(-2) = 0.135 and are trained as near-hard negatives, leaving the target a
# delta with no soft neighbourhood. 4.0 tiles is one cell, which is what the
# vanilla CenterNet baseline in this repo derives from its box size, and that
# baseline reaches a higher IR than Director despite a far weaker backbone.
DIRECTOR_RENDER_SIGMA = 4.0
# torchvision freezes layer1 at trainable_layers=3, and layer1 is the level the
# heatmap head reads. See DirectorCenterNet for why that costs this model more
# than it costs Mask R-CNN.
DIRECTOR_TRAINABLE_LAYERS = 5
# Width of the hidden convs in each head. Mask R-CNN scores a region with a
# ~14M-parameter MLP over ROIAlign-pooled features; this head is ~184k at
# width 64. Exposed so the capacity hypothesis can be tested without a code
# change - the receptive field is not the issue (two 3x3 convs already span
# 5x5 cells = 20x20 tiles, wider than the 12x20-tile viewport).
DIRECTOR_HEAD_CONV = 64
# Place a focal positive at every observer's own viewport centre, not only at
# the Top-1 mode. See _add_observer_positives in losses/director_losses.py.
DIRECTOR_DENSE_POSITIVES = False
DIRECTOR_K = 3
# The smallest meaningful mode amplitude is 1 observer out of U, i.e. 1/5 = 0.2
# (the Gaussian is rendered on its assigned grid cell, so that is exact rather
# than attenuated). tau has to sit clearly below it or a minority mode can
# never be emitted no matter how well the model fits it. inference's
# score_threshold must match; at 0.3 it was the binding cut and no auxiliary
# region could survive.
DIRECTOR_TAU = 0.1
# Auxiliary target mass below this is indistinguishable from background, and is
# left to L_hcm rather than supervised by L_rmc.
DIRECTOR_AUX_SUPPORT_FLOOR = 0.01

# L_smooth (Eq 18). Displacement is measured in tiles, so the map diagonal is
# the natural unit that puts the term on an O(1) scale next to the focal loss.
MAP_DIAGONAL_TILES = (ORIGIN_SHAPE[0] ** 2 + ORIGIN_SHAPE[1] ** 2) ** 0.5  # ~181.02
# Below this displacement the penalty stays quadratic (small camera motion is
# free); above it the penalty is linear, so a big jump is expensive but its
# gradient is bounded. ~5 tiles is the observed per-step displacement of the
# healthy ablations (VD 3.6-5.3).
DIRECTOR_SMOOTH_HUBER_DELTA = 5.0
# Epochs [0, START) train with L_smooth off, then it ramps linearly to full
# weight at FULL. Applying it from step 0 lets the model reach a frozen-camera
# solution before the heatmap has learned anything worth stabilising.
DIRECTOR_SMOOTH_WARMUP_START = 5
DIRECTOR_SMOOTH_WARMUP_FULL = 10
# Radius (in feature cells) of the softmax window used to build a heatmap-
# differentiable primary centre for L_smooth. See DirectorCenterNet.
DIRECTOR_SOFT_CENTER_RADIUS = 2
# Feature cells dropped from the outside of the heatmap when picking peaks.
# max_pool2d pads with -inf, so border cells face fewer competitors and become
# local maxima far more often than interior ones.
DIRECTOR_PEAK_BORDER_MARGIN = 1


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