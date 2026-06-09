#!/bin/bash

# 파일 경로와 변경할 문자열 설정
FILE_PATH="../../.synology_chat_webhook_url"
OLD_TEXT="baecm.xyz:5001"
NEW_TEXT="121.148.88.92:62543"

# 파일 존재 여부 확인
if [ ! -f "$FILE_PATH" ]; then
    echo "오류: 파일을 찾을 수 없습니다: $FILE_PATH"
    exit 1
fi

# sed 명령어를 사용하여 내용 변경
# -i 옵션은 파일을 직접 수정(in-place)합니다.
sed -i "s/$OLD_TEXT/$NEW_TEXT/g" "$FILE_PATH"

echo "$FILE_PATH 파일 내의 모든 '$OLD_TEXT'가 '$NEW_TEXT'(으)로 변경되었습니다."