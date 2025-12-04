#!/bin/bash
set -e

GPU=1

# density ablation (8 runs)
NVIDIA_VISIBLE_DEVICES=$GPU make run ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=density020,density040,density010,density050,density060,density070,density080,density090 \
  "
