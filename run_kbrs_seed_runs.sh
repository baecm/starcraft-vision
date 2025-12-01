#!/bin/bash
set -e

GPU=0

# kbrs 3-fold seed runs (3-fold x 3 seed ==> 9 runs)
NVIDIA_VISIBLE_DEVICES=$GPU make run ARGS="-m \
  dataset=fold1,fold2,fold3 \
  model=kbrs \
  seed=123,456,789 \
  kbrs_loss=kbrs025 \
  kbrs_score=base_score \
  "