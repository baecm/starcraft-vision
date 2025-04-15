#!/bin/bash
set -e

COMMAND="$1"
shift

echo "[Entrypoint] COMMAND = $COMMAND"
echo "[Entrypoint] ARGS    = $@"

log() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S')] $1"
}

case "$COMMAND" in
  preprocess_input)
    log "Preprocessing input..."
    exec python src/preprocessing/input.py "$@"
    ;;
  preprocess_label)
    log "Preprocessing label..."
    exec python src/preprocessing/label.py "$@"
    ;;
  preprocess_pair)
    log "Preprocessing pair..."
    exec python src/preprocessing/pair.py "$@"
    ;;
  train)
    log "Training model..."
    exec python src/train.py "$@"
    ;;
  evaluate)
    log "Evaluating model..."
    exec python src/evaluate.py "$@"
    ;;
  *)
    echo "[Error] Unknown command: $COMMAND"
    echo "Try one of: preprocess_input, preprocess_label, preprocess_pair, train, evaluate"
    exit 1
    ;;
esac
