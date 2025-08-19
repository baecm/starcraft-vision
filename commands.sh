#!/bin/bash
# Example commands for running experiments

# replays
# 36 212 438 522 1660
# 1559 1628 2351 6219 11251
# 275 1725 3613 4520 4664
# 6254
# 3529 3972 7191 7950 7970 9105 9301 9795

# Preprocess input
# make preprocess_input ARGS="--replays 36 212 438 522 1660 --include-components worker ground air building vision"
make preprocess_input ARGS="--replays 36 212 438 522 1660 --include-components worker ground air building vision neutral resource terrain"
make preprocess_input ARGS="--replays 1559 1628 2351 6219 11251 --include-components worker ground air building vision neutral resource terrain"
make preprocess_input ARGS="--replays 275 1725 3613 4520 4664 --include-components worker ground air building vision neutral resource terrain"
make preprocess_input ARGS="--replays 6254 3529 3972 7191 7950 --include-components worker ground air building vision neutral resource terrain"
make preprocess_input ARGS="--replays 7970 9105 9301 9795 --include-components worker ground air building vision neutral resource terrain"
make preprocess_input ARGS="--replays 6254 2351 275 1725 3613 4520 4664 3529 3972 7191 7950 7970 9105 9301 9795 --include-components worker ground air building vision neutral resource terrain"

# Preprocess label
make preprocess_label ARGS="--replays 36 212 438 522 1660 --method all_correct"
make preprocess_label ARGS="--replays 1559 1628 2351 6219 11251 --method all_correct"
make preprocess_label ARGS="--replays 275 1725 3613 4520 4664 --method all_correct"
make preprocess_label ARGS="--replays 6254 3529 3972 7191 7950 --method all_correct"
make preprocess_label ARGS="--replays 7970 9105 9301 9795 --method all_correct"
make preprocess_label ARGS="--replays 36 212 438 522 1660 1559 1628 2351 6219 11251 275 1725 3613 4520 4664 6254 3529 3972 7191 7950 7970 9105 9301 9795 --method all_correct"

# Preprocess pair
# make preprocess_pair ARGS="--replays 36 212 438 522 1660 --method legacy --output channel"

# Training with specific replays
# make train ARGS="--replays 36 212 438 522 1660 --label-method legacy --sample-ratio 0.1 --log-level log"

make train ARGS="--replays 36 --train-replays 36 --test-replays 6254 --label-method all_correct --max-epoch 1 --window-size 4 --batch-size 8 --sample-ratio 0.1 --log-level log --use-kbrs"
make train ARGS="--replays 36 212 438 522 1660 --label-method all_correct --max-epoch 5 --batch-size 8 --window-size 4 --sample-ratio 0.05 --log-level log"
make train ARGS="--replays 36 212 438 522 1660 --label-method all_correct --max-epoch 15 --batch-size 8 --learning-rate 0.005 --sample-ratio 1.0 --log-level log"
make train ARGS="--replays 36 212 438 522 1660 --label-method all_correct --max-epoch 15 --batch-size 8 --learning-rate 0.005 --sample-ratio 1.0 --log-level log --use-kbrs"
make train ARGS="--replays 36 212 438 522 1660 --label-method all_correct --max-epoch 15 --batch-size 8 --learning-rate 0.005 --sample-ratio 0.0001 --log-level log --use-kbrs"
# make train ARGS="--replays 36 212 438 522 1660 --label-method all_correct --max-epoch 1 --batch-size 8 --sample-ratio 0.01 --log-level log"

# # Evaluate model
# make evaluate ARGS=" \
# --set set_0
# --model-name legacy_win1_b32_20250423_060708
# --model-number 9"

# make evaluate ARGS=" \
#   --set set_0 \
#   --label-method all_correct \
#   --window-size 1 \
#   --batch-size 8 \
#   --model-number 4 \
#   --data-root /home/bcm/workspace/starcraft/data \
#   --model-root /home/bcm/workspace/starcraft/models \
#   --partial-length 0.1 \ 
#   --out-csv results.csv \
#   "

# Training
make train ARGS=" \
  --replays 36 212 438 522 1660 6254 \
  --train-replays 36 212 438 522 1660 \
  --test-replays 6254 \
  --include-components worker ground air building vision \
  --label-method all_correct \
  --max-epoch 15 \
  --window-size 4 \
  --batch-size 8 \
  --sample-ratio 1.0 \
  --log-level log \
  --use-kbrs \
  "

# Inference
make inference ARGS=" \
  --replays 36 212 438 522 1660 6254 \
  --include-components worker ground air building vision \
  --model-name all_correct_win4_b8_20250815_073611 \
  --model-number 14 \
  --window-size 4 \
  --label-method all_correct \
  --batch-size 16 \
  --score-threshold 0.0 \
  --sample-ratio 1.0 \
  --output-dir /workspace/predictions \
  "

# Evaluation
make evaluate ARGS=" \
  --replays 36 212 438 522 1660 \
  --label-method all_correct \
  --pred-names all_correct_win4_b8_20250815_073611 \
  --model-number 14 \
  --ic-thresholds 0 0.3 0.5 \
  --frame-select first \
  --dump-vpd \
  --log-level log \
  "

make evaluate ARGS=" \
  --replays 6254 \
  --label-method all_correct \
  --pred-names all_correct_win4_b8_20250815_073611 \
  --model-number 14 \
  --ic-thresholds 0 0.3 0.5 \
  --frame-select first \
  --dump-vpd \
  --log-level log \
  "