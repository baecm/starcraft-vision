#!/bin/bash
# Example commands for running experiments

# Preprocess input
make preprocess_input ARGS="--replays 36 212 438 522 1660 --include-components worker ground air building vision"

# Preprocess label
make preprocess_label ARGS="--replays 36 212 438 522 1660"

# Preprocess pair
make preprocess_pair ARGS="--replays 36 212 438 522 1660 --method legacy --output channel"

# Training with specific replays
make train ARGS="--replays 36 212 438 522 1660"

# Evaluate model
make evaluate ARGS="--replays 438 --load-dir /workspace/models/..."
