#!/bin/bash
set -e  # fail fast

echo "Generating input channels..."

python src/preprocessing/label.py "$@"

echo "Preprocessing completed."
