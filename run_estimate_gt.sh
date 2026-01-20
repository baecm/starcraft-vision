#!/bin/bash
set -e

# fold1
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode gt \
  --replays 275 1725 3613 4520 4664 \
  --skip-kbrs \
  "

# fold2
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode gt \
  --replays 1559 1628 2351 6219 11251 \
  --skip-kbrs \
  "

# fold3
NVIDIA_VISIBLE_DEVICES=0 make estimate ARGS="
  --mode gt \
  --replays 36 212 438 522 1660 \
  --skip-kbrs \
  "
