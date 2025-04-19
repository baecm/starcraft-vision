#!/bin/bash
set -e

COMMAND=$1
shift

REPLAY_IDS=()
ARGS=()

mkdir -p logs

while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--replays" ]]; then
    shift
    while [[ $# -gt 0 && "$1" != --* ]]; do
      REPLAY_IDS+=("$1")
      shift
    done
  else
    ARGS+=("$1")
    shift
  fi
done

if [[ "$COMMAND" != "preprocess_input" && "$COMMAND" != "preprocess_label" && "$COMMAND" != "preprocess_pair" ]]; then
  echo "[ERROR] Invalid command: $COMMAND"
  exit 1
fi

MAX_PARALLEL=2
if [ "$MAX_PARALLEL" -lt 1 ]; then MAX_PARALLEL=1; fi

echo "[INFO] Running '$COMMAND' for ${#REPLAY_IDS[@]} replays (max $MAX_PARALLEL in parallel)"
echo "[INFO] Shared args: ${ARGS[@]}"

CURRENT_PARALLEL=0

for REPLAY_ID in "${REPLAY_IDS[@]}"; do
  LOG_FILE="logs/${COMMAND}_${REPLAY_ID}.log"
  echo "[START] Replay $REPLAY_ID → $LOG_FILE"

  docker compose -f infra/docker-compose.yml run --rm dispatcher \
    "$COMMAND" --replays "$REPLAY_ID" "${ARGS[@]}" > "$LOG_FILE" 2>&1 &

  CURRENT_PARALLEL=$((CURRENT_PARALLEL + 1))

  if [ "$CURRENT_PARALLEL" -ge "$MAX_PARALLEL" ]; then
    wait -n  # 하나가 끝날 때까지 대기
    CURRENT_PARALLEL=$((CURRENT_PARALLEL - 1))
  fi
done

wait
echo "[DONE] All parallel tasks finished."
