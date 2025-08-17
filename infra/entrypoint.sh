#!/bin/bash
set -e

export WANDB_API_KEY=$(cat /run/secrets/wandb_api_key)
echo "[Entrypoint] WANDB_API_KEY = $WANDB_API_KEY"
export SYNOLOGY_CHAT_WEBHOOK_URL=$(cat /run/secrets/synology_chat_webhook_url)
echo "[Entrypoint] SYNOLOGY_CHAT_WEBHOOK_URL = $SYNOLOGY_CHAT_WEBHOOK_URL"

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
inference)
  log "Running inference..."
  exec python src/inference.py "$@"
  ;;
evaluate)
  log "Evaluating model..."
  exec python src/evaluate.py "$@"
  ;;
debug)
  log "Debugging..."
  exec /bin/bash "$@"
  ;;
*)
  echo "[Error] Unknown command: $COMMAND"
  echo "Try one of: preprocess_input, preprocess_label, preprocess_pair, train, inference, evaluate"
  exit 1
  ;;
esac
