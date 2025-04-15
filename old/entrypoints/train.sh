#!/bin/bash
set -e  # fail fast

echo "Training model..."

python src/main.py --train --replays "$@"

echo "Training completed."
