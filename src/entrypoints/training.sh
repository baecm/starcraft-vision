#!/bin/bash
set -e  # fail fast

echo "Training model..."

python src/main.py "$@"

echo "Training completed."
