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