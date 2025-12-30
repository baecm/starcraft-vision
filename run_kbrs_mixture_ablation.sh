#!/bin/bash
set -e

GPU=0

# mixture ablation (8 runs)
# NVIDIA_VISIBLE_DEVICES=$GPU make run ARGS="-m \
#   dataset=fold1 \
#   model=kbrs \
#   seed=123 \
#   kbrs_loss=kbrs025 \
#   kbrs_score=mixture/100,mixture/150,mixture/200,mixture/250,mixture/350,mixture/400,mixture/450,mixture/500 \
#   "

NVIDIA_VISIBLE_DEVICES=$GPU make run ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=mixture/000,mixture/100,mixture/150,mixture/450,mixture/500 \
  "
