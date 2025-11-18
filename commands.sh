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

NVIDIA_VISIBLE_DEVICES=0 make train ARGS=" \
  --replays 36 212 438 522 1660 1559 1628 2351 6219 11251 \
  --include-components worker ground air building vision \
  --label-method all_correct \
  --max-epoch 30 \
  --window-size 4 \
  --interval 8 \
  --batch-size 16 \
  --sample-ratio 0.05 \
  --log-level log \
  --use-kbrs \
  --loss-weights loss_objectness 1.0 \
  --loss-weights loss_rpn_box_reg 1.0 \
  --loss-weights loss_kbrs 0.25 \
  --kbrs-param score_weights.density=0.3 \
  --kbrs-param score_weights.mixture=3.0 \
  --kbrs-param score_weights.centeredness=0.3 \
  "

# Training with specific replays
NVIDIA_VISIBLE_DEVICES=0 make train ARGS=" \
  --replays 36 212 438 522 1660 1559 1628 2351 6219 11251 \
  --include-components worker ground air building vision \
  --label-method all_correct \
  --max-epoch 30 \
  --window-size 4 \
  --interval 8 \
  --batch-size 16 \
  --sample-ratio 0.05 \
  --log-level log \
  --use-kbrs \
  --loss-weights loss_objectness 1.0 \
  --loss-weights loss_rpn_box_reg 1.0 \
  --loss-weights loss_kbrs 0.25 \
  --kbrs-param score_weights.density=0.3 \
  --kbrs-param score_weights.mixture=3.0 \
  --kbrs-param score_weights.centeredness=0.3 \
  "

NVIDIA_VISIBLE_DEVICES=1 make inference ARGS=" \
  --replays 275 1725 3613 4520 4664 \
  --include-components worker ground air building vision \
  --model-name all_correct_win4_b16_20250823_060441 \
  --model-number 30 \
  --window-size 4 \
  --label-method all_correct \
  --batch-size 16 \
  --score-threshold 0.0 \
  --sample-ratio 1.0 \
  --output-dir /workspace/predictions \
  "

# set 2: all_correct_win4_b16_20250823_060459
NVIDIA_VISIBLE_DEVICES=1 make train ARGS=" \
  --replays 36 212 438 522 1660 275 1725 3613 4520 4664\
  --include-components worker ground air building vision \
  --label-method all_correct \
  --max-epoch 30 \
  --window-size 4 \
  --interval 8 \
  --batch-size 16 \
  --sample-ratio 1.0 \
  --log-level log \
  "

NVIDIA_VISIBLE_DEVICES=1 make inference ARGS=" \
  --replays 1559 1628 2351 6219 11251 \
  --include-components worker ground air building vision \
  --model-name all_correct_win4_b16_20250823_060459 \
  --model-number 30 \
  --window-size 4 \
  --label-method all_correct \
  --batch-size 16 \
  --score-threshold 0.0 \
  --sample-ratio 1.0 \
  --output-dir /workspace/predictions \
  "

# set 3: all_correct_win4_b16_20250823_060525
NVIDIA_VISIBLE_DEVICES=1 make train ARGS=" \
  --replays 1559 1628 2351 6219 11251 275 1725 3613 4520 4664 \
  --include-components worker ground air building vision \
  --label-method all_correct \
  --max-epoch 30 \
  --window-size 4 \
  --interval 8 \
  --batch-size 16 \
  --sample-ratio 1.0 \
  --log-level log \
  "

NVIDIA_VISIBLE_DEVICES=1 make inference ARGS=" \
  --replays 36 212 438 522 1660 \
  --include-components worker ground air building vision \
  --model-name all_correct_win4_b16_20250823_060525 \
  --model-number 30 \
  --window-size 4 \
  --label-method all_correct \
  --batch-size 16 \
  --score-threshold 0.0 \
  --sample-ratio 1.0 \
  --output-dir /workspace/predictions \
  "

# # Evaluation
# # set 1: all_correct_win4_b16_20250823_060441
# make evaluate ARGS=" \
#   --replays 275 1725 3613 4520 4664 \
#   --label-method all_correct \
#   --pred-names all_correct_win4_b16_20250823_060441 \
#   --model-number 30 \
#   --ic-thresholds 0 0.3 0.5 \
#   --frame-select all \
#   --dump-vpd \
#   --log-level log \
#   "
# # set 2: all_correct_win4_b16_20250823_060459
# make evaluate ARGS=" \
#   --replays 1559 1628 2351 6219 11251 \
#   --label-method all_correct \
#   --pred-names all_correct_win4_b16_20250823_060459 \
#   --model-number 30 \
#   --ic-thresholds 0 0.3 0.5 \
#   --frame-select first \
#   --dump-vpd \
#   --log-level log \
#   "
# # set 3: all_correct_win4_b16_20250823_060525
# make evaluate ARGS=" \
#   --replays 36 212 438 522 1660 \
#   --label-method all_correct \
#   --pred-names all_correct_win4_b16_20250823_060525 \
#   --model-number 30 \
#   --ic-thresholds 0 0.3 0.5 \
#   --frame-select first \
#   --dump-vpd \
#   --log-level log \
#   "