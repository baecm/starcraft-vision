# MODEL CONFIGURATION
DEVICE = 'cuda'
NUM_CLASSES = 2  # 0: background, 1: unit
WINDOW_SIZE = 1
INTERVAL = 1
DO_NORMALIZE = False

# TRAIN CONFIGURATION
TRAIN_BATCH_SIZE = 8
TRAIN_NUM_WORKERS = 4
TRAIN_EPOCHS = 10
TRAIN_LEARNING_RATE = 0.005
TRAIN_MOMENTUM = 0.9
TRAIN_WEIGHT_DECAY = 0.0005
TRAIN_LR_SCHEDULER_STEP_SIZE = 3
TRAIN_LR_SCHEDULER_GAMMA = 0.1
TRAIN_LOG_INTERVAL = 10
TRAIN_SAVE_INTERVAL = 1

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

from enum import Enum

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

KBRS_PARAMS = {
    'feature_map_name': 'smallest',
    'scorer_impl': 'conv',
    'detach_scorer_input': False,
    'loss_weights': {
        'loss_kbrs': 0.25,
    },
    'score_weights': {
        'density': 0.3,
        'mixture': 3.0,
        'centeredness': 0.3
    },
    'region_size': (20, 12),
    'learnable': 'static',
    'tau': 4.0,
    'use_entropy': False,
    'projections': [{'name': 'A', 'channels': [0, 1, 2, 3]},
                    {'name': 'B', 'channels': [4, 5, 6, 7]}],
    'mixture_between': ('A', 'B'),
    # 게이트(vision-like): window_size>1이면 자동 확장됨
    'gate_channels': [8],
    'gate_reduce': 'mean',
    'gate_gain': 0.8,
    # scorer 해상도/속도
    'score_stride': 1,
    'downsample_before': None,
    'log_into_losses': False,
    # 혼합도 설정(중요)
    'mixture_mode': 'confusion',     # 'confusion' | 'entropy' | 'agreement'
    'mixture_power': 2.0,            # p≈0.5 근처 강조
    # builder가 주입
    # 'window_size': window_size,
    # 'per_window': 9,
    'viz_components': True,
    'accumulate_epoch': True,
    'component_losses': True,
}