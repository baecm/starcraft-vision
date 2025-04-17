import os
import glob
import re
import numpy as np
import tqdm


def sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


class Resolver:
    UNIT_TYPE_INDEX = {
        "worker": 0,
        "ground": 1,
        "air": 2,
        "building": 3
    }

    def __init__(self, temp_dir, resolution_frame, output_dir, include_components=None):
        self.temp_dir = temp_dir
        self.resolution_frame = resolution_frame
        self.output_dir = output_dir
        self.include_components = include_components or ["worker", "ground", "air", "building", "neutral", "vision", "terrain"]

        # 구분: 유닛 타입 vs 전체 블록 선택
        self.unit_types = [k for k in self.UNIT_TYPE_INDEX if k in self.include_components]
        self.include_neutral = "neutral" in self.include_components
        self.include_vision = "vision" in self.include_components
        self.include_terrain = "terrain" in self.include_components

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _filter_unit_types(self, state):
        indices = [self.UNIT_TYPE_INDEX[t] for t in self.unit_types]
        return state[indices, :, :] if indices else np.empty((0, *state.shape[1:]), dtype=state.dtype)

    def run(self):
        temp_dir = os.path.abspath(self.temp_dir)

        state_player_1_files = sorted(glob.glob(os.path.join(temp_dir, "state_*_player_1.npy")), key=sort_key)
        state_player_2_files = sorted(glob.glob(os.path.join(temp_dir, "state_*_player_2.npy")), key=sort_key)
        state_neutral_files = sorted(glob.glob(os.path.join(temp_dir, "state_*_neutral.npy")), key=sort_key)
        vision_files = sorted(glob.glob(os.path.join(temp_dir, "vision_*.npy")), key=sort_key)
        terrain_files = glob.glob(os.path.join(temp_dir, "terrain.npy"))

        if self.include_terrain:
            if not terrain_files:
                raise FileNotFoundError("terrain.npy does not exist.")
            terrain = np.load(terrain_files[0])

        min_length = min(
            len(state_player_1_files),
            len(state_player_2_files),
            len(state_neutral_files) if self.include_neutral else float('inf'),
            len(vision_files) if self.include_vision else float('inf'),
            self.resolution_frame
        )

        for i in tqdm.tqdm(range(min_length), desc="Resolving outputs", miniters=max(1, min_length // 20)):
            result_list = []

            player_1 = self._filter_unit_types(np.load(state_player_1_files[i]))
            player_2 = self._filter_unit_types(np.load(state_player_2_files[i]))

            result_list.append(player_1)
            result_list.append(player_2)

            if self.include_neutral:
                neutral = np.load(state_neutral_files[i])
                result_list.append(neutral)

            if self.include_vision:
                vision = np.load(vision_files[i])
                result_list.append(np.expand_dims(vision, axis=0))

            if self.include_terrain:
                result_list.append(np.expand_dims(terrain, axis=0))

            result = np.vstack(result_list)
            np.save(os.path.join(self.output_dir, f"{i}.npy"), result)

            # 파일 제거
            os.remove(state_player_1_files[i])
            os.remove(state_player_2_files[i])
            if self.include_neutral:
                os.remove(state_neutral_files[i])
            if self.include_vision:
                os.remove(vision_files[i])
