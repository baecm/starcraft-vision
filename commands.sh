#!/bin/bash
# Example commands for running experiments

# Preprocess input
make preprocess_input ARGS="--replays 36 212 438 522 1660 --include-components worker ground air building vision"

# Preprocess label
make preprocess_label ARGS="--replays 36 212 438 522 1660 --method all_correct"

# Preprocess pair
make preprocess_pair ARGS="--replays 36 212 438 522 1660 --method legacy --output channel"

# Training with specific replays
make train ARGS="--replays 36 212 438 522 1660 --label-method legacy"
make train ARGS="--replays 36 212 438 522 1660 --label-method legacy --sample-ratio 0.1"
make train ARGS="--replays 36 212 438 522 1660 --label-method legacy --sample-ratio 0.1 --log-level log"

make train ARGS="--replays 36 212 438 522 1660 --label-method all_correct --max-epoch 5 --batch-size 8 --sample-ratio 0.1 --log-level log"

# Evaluate model
make evaluate ARGS=" \
--set set_0
--model-name legacy_win1_b32_20250423_060708
--model-number 9"

# Inference
make inference ARGS=" \
  --replays 36 212 438 522 1660 \
  --model-name all_correct_win1_b8_20250530_04111 \
  --model-number 4 \
  --label-method all_correct \
  --batch-size 8 \
  --score-thr 0.5 \
  --output-dir /workspace/predictions \
  "