#!/bin/bash
set -e

# 0.25
# # baseline (kbrs_fold1_s123)
# NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
#   --mode model \
#   --model-name all_correct_win4_kbrs_fold1_s123_kbrs025_base_score_20251219_080334 \
#   --epoch 30 \
#   --replays 275 1725 3613 4520 4664 \
#   --skip-kbrs \
#   "

# 0.5
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_kbrs_fold1_s123_kbrs050_base_score_20251204_075521 \
  --epoch 30 \
  --replays 275 1725 3613 4520 4664 \
  --skip-kbrs \
  "

# 0.75
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_kbrs_fold1_s123_kbrs075_base_score_20251215_013131 \
  --epoch 30 \
  --replays 275 1725 3613 4520 4664 \
  --skip-kbrs \
  "

# 1.0
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_kbrs_fold1_s123_kbrs100_base_score_20251218_144314 \
  --epoch 30 \
  --replays 275 1725 3613 4520 4664 \
  --skip-kbrs \
  "
