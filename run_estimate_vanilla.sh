#!/bin/bash
set -e

# fold1
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_kbrs_fold1_s123_kbrs025_base_score_20251204_075537 \
  --epoch 30 \
  --replays 275 1725 3613 4520 4664 \
  --skip-kbrs \
  "

NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_vanilla_fold1_s456_20251202_003952 \
  --epoch 30 \
  --replays 275 1725 3613 4520 4664 \
  --skip-kbrs \
  "

NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_vanilla_fold1_s789_20251202_163217 \
  --epoch 30 \
  --replays 275 1725 3613 4520 4664 \
  --skip-kbrs \
  "

# fold2
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_vanilla_fold2_s123_20251203_081925 \
  --epoch 30 \
  --replays 1559 1628 2351 6219 11251 \
  --skip-kbrs \
  "

NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_vanilla_fold2_s456_20251204_002252 \
  --epoch 30 \
  --replays 1559 1628 2351 6219 11251 \
  --skip-kbrs \
  "

NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_vanilla_fold2_s789_20251204_163313 \
  --epoch 30 \
  --replays 1559 1628 2351 6219 11251 \
  --skip-kbrs \
  "

# fold3
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_vanilla_fold3_s123_20251205_081306 \
  --epoch 30 \
  --replays 36 212 438 522 1660 \
  --skip-kbrs \
  "

NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_vanilla_fold3_s456_20251205_235535 \
  --epoch 30 \
  --replays 36 212 438 522 1660 \
  --skip-kbrs \
  "
  
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_vanilla_fold3_s789_20251206_161212 \
  --epoch 30 \
  --replays 36 212 438 522 1660 \
  --skip-kbrs \
  "