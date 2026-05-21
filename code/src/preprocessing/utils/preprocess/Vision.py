import os
import numpy as np
import tqdm


class Vision:
    def __init__(self, data):
        self.data = data
        self.pivot = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def run_length_decode(self, data: str):
        """
        Decode run-length encoded vision data.
        """
        parts = data.split(",")
        frame = int(parts[0])
        first_val = int(parts[1])
        counts = list(map(int, parts[2].split()))

        result = []
        current_val = first_val

        for count in counts:
            result.extend([current_val] * count)
            current_val = 1 - current_val

        self.save_vision(frame, result)

    def save_vision(self, frame, vision):
        assert self.pivot < frame, f"Frame {frame} is out of order. Current pivot: {self.pivot}"

        for i in range(self.pivot, frame):
            vision = np.array(vision).reshape((self.data["height"], self.data["width"]))
            np.save(os.path.join(self.data["temp_dir"], f"vision_{i}.npy"), vision)

        self.pivot = frame

    def run(self):
        total = len(self.data["vision_raw"])
        for line in tqdm.tqdm(
            self.data["vision_raw"],
            desc="Decode vision...",
            miniters=max(1, total // 20)
        ):
            self.run_length_decode(line)

        del self.data["vision_raw"]
