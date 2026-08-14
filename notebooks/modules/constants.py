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
    
GT_COLS = [f'gt_{i}' for i in range(5)]

GRID_W = 128
GRID_H = 128
KW = 20      # 커널 너비 (kw)
KH = 12      # 커널 높이 (kh)
STRIDE = 1