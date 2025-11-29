#!/bin/bash
set -e

GPU=0

NVIDIA_VISIBLE_DEVICES=$GPU make train ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs025,kbrs050,kbrs075,kbrs100 \
  kbrs_score=base_score"

NVIDIA_VISIBLE_DEVICES=$GPU make train ARGS="-m \
  dataset=fold1 \
  model=kbrs \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=density010,density020,density040,density050,density060,density070,density080,density090"
