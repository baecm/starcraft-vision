#!/usr/bin/env bash
set -ex  # 에러 발생 시 즉시 중단하고, 실행되는 명령어를 로그에 출력

export PYTHONPATH=/code

# 1. 결과물을 저장할 디렉토리 생성
echo "Setting up results directories..."
mkdir -p /results/models /results/logs /results/predictions

# 2. 실행 (Makefile을 거치지 않고 entrypoint.sh를 직접 호출)
# 주의: 아래 명령어는 예시이며, 실제 논문에서 보여주고자 하는 대표적인 실험 하나를 선택해야 합니다.
echo "Running the main training & inference pipeline..."

# 예시: 기존 make run ARGS="..." 와 동일한 효과를 내는 직접 실행 명령어
bash ./entrypoint.sh run \
  -m \
  dataset=fold1 \
  model=kbrs \
  mode=train_and_inference \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=base \
  kbrs_score.density=0.3 \
  kbrs_score.mixture=3.0 \
  kbrs_score.centeredness=0.3

echo "Run completed successfully. Results are saved in /results."