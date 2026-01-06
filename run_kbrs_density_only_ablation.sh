#!/bin/bash
set -e

GPU=1

# density ablation (8 runs)
NVIDIA_VISIBLE_DEVICES=$GPU make run ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=density_only/010,density_only/020,density_only/040,density_only/050 \
  "

# NVIDIA_VISIBLE_DEVICES=$GPU make run ARGS="-m \
#   dataset=fold1 \
#   model=kbrs \
#   seed=123 \
#   kbrs_loss=kbrs025 \
#   kbrs_score=density_only/060,density_only/070,density_only/080,density_only/090 \
#   "
