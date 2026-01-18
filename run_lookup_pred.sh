#!/bin/bash
set -e

# fold1: 275 1725 3613 4520 4664
# fold2: 1559 1628 2351 6219 11251
# fold3: 36 212 438 522 1660

NVIDIA_VISIBLE_DEVICES=0 make lookup ARGS=" \
  --replays 275 1725 3613 4520 4664 1559 1628 2351 6219 11251 36 212 438 522 1660 \
  --label-source pred \
  --id-string all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_500_20260117_032417 \
  --epoch 30 \
  --label-method all_correct \
  --skip-missing-npz \
  --force-row-on-error \
  --num-workers 16 \
  --log-level log \
  "