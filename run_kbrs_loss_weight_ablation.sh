#!/bin/bash
set -e

GPU=0

# kbrs loss weight ablation (4 runs)
NVIDIA_VISIBLE_DEVICES=$GPU make run ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs050,kbrs075,kbrs100 \
  kbrs_score=base_score \
  "