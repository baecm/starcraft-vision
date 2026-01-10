#!/bin/bash
set -e

# fold1
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_kbrs_fold1_s123_kbrs025_base_score_20251219_080334 \
  --epoch 30 \
  --replays 275 1725 3613 4520 4664 \
  --skip-kbrs \
  "

NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_kbrs_fold1_s456_kbrs025_base_score_20260101_175816 \
  --epoch 30 \
  --replays 275 1725 3613 4520 4664 \
  --skip-kbrs \
  "

NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_kbrs_fold1_s789_kbrs025_base_score_20251230_004200 \
  --epoch 30 \
  --replays 275 1725 3613 4520 4664 \
  --skip-kbrs \
  "

# fold2
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_kbrs_fold2_s123_kbrs025_base_score_20251222_080550 \
  --epoch 30 \
  --replays 1559 1628 2351 6219 11251 \
  --skip-kbrs \
  "

NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_kbrs_fold2_s456_kbrs025_base_score_20251231_031613 \
  --epoch 30 \
  --replays 1559 1628 2351 6219 11251 \
  --skip-kbrs \
  "

NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_kbrs_fold2_s789_kbrs025_base_score_20251231_215319 \
  --epoch 30 \
  --replays 1559 1628 2351 6219 11251 \
  --skip-kbrs \
  "

# fold3
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_kbrs_fold3_s123_kbrs025_base_score_20251230_083127 \
  --epoch 30 \
  --replays 36 212 438 522 1660 \
  --skip-kbrs \
  "

NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_kbrs_fold3_s456_kbrs025_base_score_20251231_151012 \
  --epoch 30 \
  --replays 36 212 438 522 1660 \
  --skip-kbrs \
  "
  
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_kbrs_fold3_s789_kbrs025_base_score_20260102_163643 \
  --epoch 30 \
  --replays 36 212 438 522 1660 \
  --skip-kbrs \
  "