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

# 단일 run 예시
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="dataset=fold1 model=kbrs seed=123 kbrs_loss=kbrs025 kbrs_score=base"

# 멀티런 예시
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="-m dataset=fold1,fold2,fold3 model=kbrs seed=123,456,789 kbrs_loss=kbrs025 kbrs_score=base"

# seed sweep
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="-m dataset=fold1 model=kbrs seed=123,456,789 kbrs_loss=kbrs025 kbrs_score=base"

# fold1, kbrs, seed=123, kbrs_score=base 고정하고 loss weight 스윕
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs025,kbrs050,kbrs075,kbrs100 \
  kbrs_score=base"

# density ablation
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=base,density010,density020,density040,density050,density060,density070,density080,density090"

# mixture ablation
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=mixture100,mixture150,mixture200,mixture250,mixture350,mixture400,mixture450,mixture500,mixture150,mixture200,mixture250,mixture350"

# centeredness ablation
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=centeredness010,centeredness020,centeredness040,centeredness050,centeredness060,centeredness070,centeredness080,centeredness090"

NVIDIA_VISIBLE_DEVICES=0 make run ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  mode=train_and_inference \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=centeredness050 \
"

make estimate ARGS=" \
  --replays 36 212 438 522 1660 1559 1628 2351 6219 11251 275 1725 3613 4520 4664 6254 3529 3972 7191 7950 7970 9105 9301 9795 \
  --mode=gt \
"

make estimate ARGS=" \
  --replays 36 \
  --mode=gt \
  --skip-kbrs \
"

# fold1
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode model \
  --model-name all_correct_win4_vanilla_fold1_s123_20251201_072032 \
  --epoch 30 \
  --replays 275 1725 3613 4520 4664 \
  --skip-kbrs \
  "

# fold2
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
  --model-name all_correct_win4_kbrs_fold3_s456_kbrs025_base_score_20251231_151012 \
  --epoch 30 \
  --replays 36 212 438 522 1660 \
  --skip-kbrs \
  "

make precheck ARGS="
  --root_dir /workspace/data/label/dst \
  --label_method all_correct \
  --replays 36,212,438,522,1660,1559,1628,2351,6219,11251,275,1725,3613,4520,4664 \
  --verbose \
  "

make precheck ARGS="
  --root_dir /workspace/data/input/dst \
  --replays 36,212,438,522,1660,1559,1628,2351,6219,11251,275,1725,3613,4520,4664 \
  --mmap \
  --max_abs_warn 1000000 \
  --workers 32 \
  --chunksize 64 \
  "

# python kbrs_cli.py cache --replays 275 3613 --num-workers 16
# fold1: 275 1725 3613 4520 4664
# fold2: 1559 1628 2351 6219 11251
# fold3: 36 212 438 522 1660
NVIDIA_VISIBLE_DEVICES=0 make cache ARGS=" \
  cache \
  --replays 275 1725 3613 4520 4664 1559 1628 2351 6219 11251 36 212 438 522 1660 \
  --num-workers 16 \
  --sample-ratio 1.0 \
  --log-level none \
"

NVIDIA_VISIBLE_DEVICES=0 make lookup ARGS=" \
  --replays 1628 \
  --label-source gt \
  --label-method all_correct \
  --skip-missing-npz \
  --force-row-on-error \
  --num-workers 16 \
  --log-level log \
  "

NVIDIA_VISIBLE_DEVICES=0 make lookup ARGS=" \
  --replays 1628 \
  --label-source pred \
  --id-string all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_500_20260117_032417 \
  --epoch 30 \
  --label-method all_correct \
  --skip-missing-npz \
  --force-row-on-error \
  --num-workers 16 \
  --log-level log \
  "