#!/bin/bash
set -e

GPU=1

# density ablation (8 runs)
NVIDIA_VISIBLE_DEVICES=$GPU make run ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=density/000,density/010,density/020,density/040,density/050,density/060,density/070,density/080,density/090 \
  "
