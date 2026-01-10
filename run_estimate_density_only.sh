#!/bin/bash
set -e

# 0.1
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_kbrs_fold1_s123_kbrs025_density_only_010_20260106_020536 \
  --epoch 30 \
  --replays 275 1725 3613 4520 4664 \
  --skip-kbrs \
  "


# 0.3
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_kbrs_fold1_s123_kbrs025_density_only_030_20260105_055837 \
  --epoch 30 \
  --replays 275 1725 3613 4520 4664 \
  --skip-kbrs \
  "

