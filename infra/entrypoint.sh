#!/bin/bash
set -e

if [ -f /run/secrets/wandb_api_key ]; then
  export WANDB_API_KEY=$(cat /run/secrets/wandb_api_key 2>/dev/null || echo "")
fi
if [ -f /run/secrets/synology_chat_webhook_url ]; then
  export SYNOLOGY_CHAT_WEBHOOK_URL=$(cat /run/secrets/synology_chat_webhook_url 2>/dev/null || echo "")
fi

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
estimate)
  log "Estimating labels..."
  exec python src/estimate.py "$@"
  ;;
run)
  log "Runninng train/inference squentially..."
  exec python src/pipeline.py "$@"
  ;;
cache)
  log "Caching replays..."
  exec python src/kbrs_cache.py "$@"
  ;;
lookup)
  log "Looking up replays..."
  exec python src/kbrs_lookup.py "$@"
  ;;
precheck)
  log "Prechecking replays..."
  exec python src/precheck.py "$@"
  ;;
profile_kbrs)
  log "Profiling KBRS..."
  exec python src/profile_kbrs_scorer.py "$@"
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
