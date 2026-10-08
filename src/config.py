"""
Project-wide constants, and the short aliases accepted on the command line.

Sections:
  map and input      tile map size, input channels and component groups
  labels             label methods; the observers' viewport size
  ranked modes       mode extraction parameters (metrics.modes.extract_modes)
  ROCI               the target augmentation of Joo et al. (2023)
  training           gradient clipping
  Director-CenterNet defaults of the multi-region heatmap model
  CLI aliases        resolve_cli_aliases

Hydra (conf/) sets the run configuration; these are the defaults it falls
back to and the constants it does not expose.
"""
import sys
from enum import Enum

# ---------------------------------------------------------------------------
# Map and input
# ---------------------------------------------------------------------------

ORIGIN_SHAPE = (128, 128)  # the game map as a tile grid (square, so the order does not matter)
TILE_SIZE = 32             # pixels per tile in the replay's own coordinates

# Input channels per frame (the order of the preprocessed .npy arrays) and the
# component names that select them (--include-components, include_components).
# They are defined at the end of this file: Channel, COMPONENT_CHANNEL_MAP.

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

# How preprocessing turns each observer's camera trace into a per-frame
# viewport label; every reported run uses all_correct.
LABEL_METHODS = [
    'legacy',
    'consider_previous',
    'unique_local_maximums',
    'all_correct',
]

# Preprocessing output formats (src/preprocessing).
OUTPUT_TYPES = ['coord', 'channel']

# The observers' viewport in tiles. Both orders are kept because both are in
# use: KERNEL_SHAPE (width, height) by preprocessing, VIEWPORT_SIZE_HW
# (height, width) by the models, the dataset and the analysis.
KERNEL_SHAPE = (20, 12)
VIEWPORT_SIZE_HW = (12, 20)
NUM_OBSERVERS_U = 5  # spectators per replay

# ---------------------------------------------------------------------------
# Ranked attention modes (metrics.modes.extract_modes)
# ---------------------------------------------------------------------------

MODE_EXTRACTION_SIGMA = 4.0          # Gaussian smoothing of the coverage map, tiles
MODE_EXTRACTION_MIN_SEP = 12.0       # minimum separation D between modes, tiles
MODE_EXTRACTION_REL_THRESHOLD = 0.35  # peak floor theta, relative to the frame's maximum
MODE_EXTRACTION_MAX_MODES = 5

# ---------------------------------------------------------------------------
# ROCI (Region of Common Interest) target augmentation
# ---------------------------------------------------------------------------
# Joo et al. (2023) add the regions the observers agree on as extra detection
# targets. Off by default: turning it on changes the labels, so the generated
# file is named "<method>_roci" and a model trained on it must be trained from
# scratch - an existing checkpoint cannot be converted.
#
# These follow their released code, not the paper's prose. The blur kernel is
# about the size of a viewport, which is what keeps the coverage map on its
# original scale - a tile two observers both cover still reads near 2 after
# blurring - and that is what makes an absolute floor just above 1 separate
# common interest from a lone observer. Widening the blur flattens the scale
# and the same floor then admits nothing at all.
ROCI_ENABLED = False
ROCI_BLUR_KSIZE = (19, 11)   # (width, height), cv2 order; sigma derived from it
ROCI_MIN_DISTANCE = 7        # tiles between peaks
ROCI_THRESHOLD = 1.1         # absolute, on the blurred coverage
ROCI_MAX_REGIONS = 0         # 0 = unbounded, as in their code

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

# Global gradient-norm clip applied in the training loop (train.py passes it to
# detection/engine_safe.py) for every model.
GRAD_CLIP_NORM = 2.0

# ---------------------------------------------------------------------------
# Director-CenterNet
# ---------------------------------------------------------------------------

# Gaussian width for the rendered targets, in tiles. The heatmap grid samples
# every `down_ratio` tiles, so 2.0 tiles is half a cell: adjacent cells fall to
# exp(-2) = 0.135 and are trained as near-hard negatives, leaving the target a
# delta with no soft neighborhood. 4.0 tiles is one cell, which is what the
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
# Place a focal positive at every observer's own viewport center, not only at
# the Top-1 mode. See _add_observer_positives in losses/director_losses.py.
DIRECTOR_DENSE_POSITIVES = False
# Which target the negative term of L_hcm is weighted by, (1 - Y)^beta:
#   "joint"   Y_all = max(Y1, Y_minus), so the auxiliary modes are protected
#             from suppression (the default; every reported run)
#   "primary" Y1, the primary mode's own Gaussian, as CornerNet's focal loss
#             does for a single object; the auxiliary modes are then ordinary
#             negatives and no cells are excluded from L_hcm
# See human_consensus_match_loss in losses/director_losses.py.
DIRECTOR_HCM_NEGATIVE_TARGET = "joint"
DIRECTOR_K = 3  # regions decoded per frame (primary + auxiliaries)
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
# solution before the heatmap has learned anything worth stabilizing.
DIRECTOR_SMOOTH_WARMUP_START = 5
DIRECTOR_SMOOTH_WARMUP_FULL = 10
# Radius (in feature cells) of the softmax window used to build a heatmap-
# differentiable primary center for L_smooth. See DirectorCenterNet.
DIRECTOR_SOFT_CENTER_RADIUS = 2
# Feature cells dropped from the outside of the heatmap when picking peaks.
# max_pool2d pads with -inf, so border cells face fewer competitors and become
# local maxima far more often than interior ones.
DIRECTOR_PEAK_BORDER_MARGIN = 1

# ---------------------------------------------------------------------------
# Input channels
# ---------------------------------------------------------------------------


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

# ---------------------------------------------------------------------------
# CLI aliases
# ---------------------------------------------------------------------------

# Short forms accepted by make train / make run (Hydra) and the inference CLI.
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