#!/bin/bash
set -e

GPU=0

# centeredness ablation (8 runs)
NVIDIA_VISIBLE_DEVICES=$GPU make train ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=centeredness010,centeredness020,centeredness040,centeredness050,centeredness060,centeredness070,centeredness080,centeredness090 \
  "
