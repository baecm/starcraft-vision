import os
# STORAGE_DIR = "Z:\\starcraft-vision\\"
STORAGE_DIR = "/workspace"

DATA_DIR = os.path.join(STORAGE_DIR, "data")
GROUND_TRUTH_DIR = os.path.join(DATA_DIR, "label", "dst")
PREDICTIONS_DIR = os.path.join(STORAGE_DIR, "predictions")