# src/config.py
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