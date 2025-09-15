#!/bin/bash

TOKEN=$(python3 - <<'PY'
import secrets; print(secrets.token_urlsafe(24))
PY
)
echo "JUPYTER_TOKEN=$TOKEN"
# 파일 경로
FILE="$HOME/.jupyter_token"

# 인자로 받은 값 저장 및 권한 변경
echo "$TOKEN" > "$FILE"
chmod 444 "$FILE"

# 저장된 값 출력
cat "$FILE"
