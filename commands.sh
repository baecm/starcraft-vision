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
"