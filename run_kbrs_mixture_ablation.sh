#!/bin/bash
set -e

GPU=3

# mixture ablation (8 runs)
NVIDIA_VISIBLE_DEVICES=$GPU make run ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=mixture250,mixture350,mixture200,mixture400,mixture100,mixture450,mixture150,mixture500 \
  "
