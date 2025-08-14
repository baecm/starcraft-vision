#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <webhook_url>"
    exit 1
fi

# 파일 경로
FILE="$HOME/.synology_chat_webhook_url"

# 인자로 받은 값 저장 및 권한 변경
echo "$1" > "$FILE"
chmod 444 "$FILE"

# 저장된 값 출력
cat "$FILE"
