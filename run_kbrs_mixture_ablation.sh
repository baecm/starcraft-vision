#!/bin/bash
set -e

GPU=0

# mixture ablation (8 runs)
NVIDIA_VISIBLE_DEVICES=$GPU make run ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=mixture100,mixture150,mixture200,mixture250,mixture350,mixture400,mixture450,mixture500 \
  "
