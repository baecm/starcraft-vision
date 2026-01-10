#!/bin/bash
set -e

# 0.0
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_000_20260102_074149 \
  --epoch 30 \
  --replays 275 1725 3613 4520 4664 \
  --skip-kbrs \
  "

# 0.1
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_010_20251216_063600 \
  --epoch 30 \
  --replays 275 1725 3613 4520 4664 \
  --skip-kbrs \
  "

# 0.2
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_020_20251204_074940 \
  --epoch 30 \
  --replays 275 1725 3613 4520 4664 \
  --skip-kbrs \
  "

# 0.3
# NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
#   --mode model \
#   --model-name all_correct_win4_kbrs_fold1_s123_kbrs025_base_score_20251219_080334 \
#   --epoch 30 \
#   --replays 275 1725 3613 4520 4664 \
#   --skip-kbrs \
#   "

# 0.4
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_040_20251224_064547 \
  --epoch 30 \
  --replays 275 1725 3613 4520 4664 \
  --skip-kbrs \
  "

# 0.5
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_050_20251219_101225 \
  --epoch 30 \
  --replays 275 1725 3613 4520 4664 \
  --skip-kbrs \
  "
