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

# Training with specific replays
make train ARGS=" \
  --replays 36 212 438 522 1660 6254 \
  --train-replays 36 212 438 522 1660 \
  --test-replays 6254 \
  --include-components worker ground air building vision \
  --label-method all_correct \
  --max-epoch 15 \
  --window-size 4 \
  --interval 8 \
  --batch-size 8 \
  --sample-ratio 1.0 \
  --log-level log \
  --use-kbrs \
  "

# Inference #6254
make inference ARGS=" \
  --replays 36 212 438 522 1660 \
  --include-components worker ground air building vision \
  --model-name all_correct_win4_b16_kbrs_20250819_075939 \
  --model-number 9 \
  --window-size 4 \
  --label-method all_correct \
  --batch-size 8 \
  --score-threshold 0.0 \
  --sample-ratio 1.0 \
  --output-dir /workspace/predictions \
  "
make inference ARGS=" \
  --replays 1559 1628 2351 6219 11251 \
  --include-components worker ground air building vision \
  --model-name all_correct_win4_b16_kbrs_20250819_075939 \
  --model-number 9 \
  --window-size 4 \
  --label-method all_correct \
  --batch-size 8 \
  --score-threshold 0.0 \
  --sample-ratio 1.0 \
  --output-dir /workspace/predictions \
  "
make inference ARGS=" \
  --replays 275 1725 3613 4520 4664 \
  --include-components worker ground air building vision \
  --model-name all_correct_win4_b16_kbrs_20250819_075939 \
  --model-number 9 \
  --window-size 4 \
  --label-method all_correct \
  --batch-size 8 \
  --score-threshold 0.0 \
  --sample-ratio 1.0 \
  --output-dir /workspace/predictions \
  "
make inference ARGS=" \
  --replays 1559 1628 2351 6219 11251 275 1725 3613 4520 4664 \
  --include-components worker ground air building vision \
  --model-name all_correct_win4_b16_20250812_062928 \
  --model-number 9 \
  --window-size 4 \
  --label-method all_correct \
  --batch-size 8 \
  --score-threshold 0.0 \
  --sample-ratio 1.0 \
  --output-dir /workspace/predictions \
  "
  
# Evaluation
make evaluate ARGS=" \
  --replays 36 212 438 522 1660 \
  --label-method all_correct \
  --pred-names all_correct_win4_b16_20250812_062928 \
  --model-number 9 \
  --ic-thresholds 0 0.3 0.5 \
  --frame-select first \
  --dump-vpd \
  --log-level log \
  "
make evaluate ARGS=" \
  --replays 1559 1628 2351 6219 11251 \
  --label-method all_correct \
  --pred-names all_correct_win4_b16_20250812_062928 \
  --model-number 9 \
  --ic-thresholds 0 0.3 0.5 \
  --frame-select first \
  --dump-vpd \
  --log-level log \
  "
make evaluate ARGS=" \
  --replays 275 1725 3613 4520 4664 \
  --label-method all_correct \
  --pred-names all_correct_win4_b16_20250812_062928 \
  --model-number 9 \
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