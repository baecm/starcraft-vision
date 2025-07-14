# MODEL CONFIGURATION
DEVICE = "cuda"
NUM_CLASSES = 2  # 0: background, 1: unit
WINDOW_SIZE = 1
DO_NORMALIZE = False
USE_KBRS = True  # KBRS Loss 사용 여부
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
