#!/bin/bash
set -e

GPU=0

# centeredness ablation (8 runs)
NVIDIA_VISIBLE_DEVICES=$GPU make run ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=centeredness_only/010,centeredness_only/020,centeredness_only/040,centeredness_only/050 \
  "

# NVIDIA_VISIBLE_DEVICES=$GPU make run ARGS="-m \
#   dataset=fold1 \
#   model=kbrs \
#   seed=123 \
#   kbrs_loss=kbrs025 \
#   kbrs_score=centeredness_only/060,centeredness_only/070,centeredness_only/080,centeredness_only/090 \
#   "
