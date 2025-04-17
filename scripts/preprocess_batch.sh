#!/bin/bash
set -e

COMMAND=$1
shift

REPLAY_IDS=()
ARGS=()

# 로그 디렉토리 준비
mkdir -p logs

# --replays 인자 분리
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

# 유효성 검사
if [[ "$COMMAND" != "preprocess_input" && "$COMMAND" != "preprocess_label" && "$COMMAND" != "preprocess_pair" ]]; then
  echo "[ERROR] Invalid command: $COMMAND"
  exit 1
fi

# 병렬 실행 제한 수 설정 (CPU 수 절반, 최소 1개)
MAX_PARALLEL=2
if [ "$MAX_PARALLEL" -lt 1 ]; then MAX_PARALLEL=1; fi

echo "[INFO] Running '$COMMAND' for ${#REPLAY_IDS[@]} replays (max $MAX_PARALLEL in parallel)"
echo "[INFO] Shared args: ${ARGS[@]}"

# 실행 카운터
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

# 남은 작업 대기
wait
echo "[DONE] All parallel tasks finished."
