#!/bin/bash
set -e

GPU=0

# vanilla 3-fold seed runs (3-fold x 3 seed ==> 9 runs)
NVIDIA_VISIBLE_DEVICES=$GPU make train ARGS="-m \
  dataset=fold1,fold2,fold3 \
  model=vanilla \
  seed=123,456,789 \
  "
