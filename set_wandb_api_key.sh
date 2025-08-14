#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <wandb_api_key>"
    exit 1
fi

# 파일 경로
FILE="$HOME/.wandb_api_key"

# 인자로 받은 값 저장 및 권한 변경
echo "$1" > "$FILE"
chmod 444 "$FILE"

# 저장된 값 출력
cat "$FILE"
