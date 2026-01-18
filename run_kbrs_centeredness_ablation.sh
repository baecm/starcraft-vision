#!/bin/bash
set -e

GPU=0

# centeredness ablation (8 runs)
NVIDIA_VISIBLE_DEVICES=$GPU make run ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=centeredness/000,centeredness/010,centeredness/020,centeredness/040,centeredness/050,centeredness/060,centeredness/070,centeredness/080,centeredness/090 \
  "
