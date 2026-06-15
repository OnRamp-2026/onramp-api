#!/usr/bin/env bash
# on-demand GPU 리랭커 내리기: Redis 키 삭제(→즉시 vector 폴백) → VESSL Workspace terminate(과금 중단).
# 사용: ./down.sh
# 환경변수: VESSL_WS_NAME=onramp-reranker
set -euo pipefail

WS_NAME="${VESSL_WS_NAME:-onramp-reranker}"
DIR="$(cd "$(dirname "$0")" && pwd)"

command -v vesslctl >/dev/null || { echo "vesslctl 미설치" >&2; exit 1; }
vesslctl auth status >/dev/null 2>&1 || vesslctl auth login

# 1) 먼저 Redis 키 삭제 — onramp-api가 죽은 URL을 더 호출하지 않게(폴백 먼저, 그다음 terminate).
"$DIR/clear.sh" || echo "  (Redis 정리 실패 — 수동 확인 필요)" >&2

# 2) 워크스페이스 종료(과금 중단). 이름→slug 해석 후 -y 로 무확인 terminate.
SLUG="$(vesslctl workspace list -o json 2>/dev/null \
        | jq -r --arg n "$WS_NAME" '.[] | select(.name==$n) | (.slug // .name)' | head -1)"
if [ -n "$SLUG" ]; then
  echo "워크스페이스 종료: $SLUG"
  vesslctl workspace terminate "$SLUG" -y
  echo "✓ GPU 리랭커 OFF (과금 중단)"
else
  echo "✓ 실행 중인 '$WS_NAME' 워크스페이스 없음 (이미 내려감)"
fi
