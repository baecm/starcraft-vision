# MODEL CONFIGURATION
DEVICE = "cuda"
NUM_CLASSES = 2  # 0: background, 1: unit
WINDOW_SIZE = 1
DO_NORMALIZE = False
KBRS_PARAMS = {
    "weights": {"density": 1.0, "mixture": 0.7, "centeredness": 1.2},
    "loss_weight": 0.5,
    "region_size": (20, 12),
    "feature_map_name": "pool"  # FPN을 사용하므로 'pool' 특징맵 사용
}

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
    "legacy",
    "consider_previous",
    "unique_local_maximums",
    "all_correct",
]

OUTPUT_TYPES = ["coord", "channel"]

KERNEL_SHAPE = (20, 12)
ORIGIN_SHAPE = (128, 128)
TILE_SIZE = 32
INTERVAL = 1

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
