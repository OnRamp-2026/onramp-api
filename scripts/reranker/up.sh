#!/usr/bin/env bash
# on-demand GPU 리랭커 켜기: VESSL Workspace(L40S) 생성 → uvicorn 자동기동(init-script)
# → 노출 URL 파싱 → /health/ready 확인 → Redis에 등록(set-url.sh).
# 끝나면 반드시 ./down.sh 로 내려서 크레딧 절약($1.80/hr).
#
# 사용: ./up.sh
# 환경변수(override 가능):
#   VESSL_WS_NAME=onramp-reranker
#   VESSL_CLUSTER=cluster-betelgeuse
#   VESSL_SPEC=resourcespec-eaac8rxxpz6j      # L40S ×1 ($1.80/hr)
#   VESSL_IMAGE=ghcr.io/onramp-2026/onramp-reranker:gpu-v1
#   VESSL_SSHKEY=sshkey-umdldri4c2ha
#   RERANKER_URL_OVERRIDE=...                 # show JSON 파싱 실패 시 URL 직접 지정
set -euo pipefail

WS_NAME="${VESSL_WS_NAME:-onramp-reranker}"
CLUSTER="${VESSL_CLUSTER:-cluster-betelgeuse}"
SPEC="${VESSL_SPEC:-resourcespec-eaac8rxxpz6j}"
IMAGE="${VESSL_IMAGE:-ghcr.io/onramp-2026/onramp-reranker:gpu-v1}"
SSHKEY="${VESSL_SSHKEY:-sshkey-umdldri4c2ha}"
# VESSL은 이미지 CMD를 실행하지 않으므로 uvicorn을 init-script에서 백그라운드 기동.
INIT_SCRIPT='cd /app && nohup uvicorn app.main:app --host 0.0.0.0 --port 8080 > /tmp/reranker.log 2>&1 &'

DIR="$(cd "$(dirname "$0")" && pwd)"

# ── preflight ───────────────────────────────────────────────────────────
command -v vesslctl >/dev/null || { echo "vesslctl 미설치" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq 미설치" >&2; exit 1; }
command -v kubectl >/dev/null || { echo "kubectl 미설치" >&2; exit 1; }
# 토큰 만료면 자동 로그인(브라우저 OAuth) — 사용자가 선택한 동작.
vesslctl auth status >/dev/null 2>&1 || vesslctl auth login

# 노출 URL 추출: 스키마 변동에 robust하게 https VESSL 엔드포인트를 재귀 스캔(reranker/8080 우선).
extract_url() {
  jq -r '[.. | strings | select(test("^https://[^ ]*vessl\\.ai"))]
         | (map(select(test("reranker|8080"))) + .) | first // empty' 2>/dev/null
}

# ── 1) 워크스페이스 생성(또는 기존 재사용) ───────────────────────────────
SLUG="$(vesslctl workspace list -o json 2>/dev/null \
        | jq -r --arg n "$WS_NAME" '.[] | select(.name==$n) | (.slug // .name)' | head -1)"
if [ -n "$SLUG" ]; then
  echo "[1/3] 기존 워크스페이스 재사용: $SLUG"
else
  echo "[1/3] 워크스페이스 생성: $WS_NAME (L40S, $IMAGE)"
  CREATE="$(vesslctl workspace create \
    --name "$WS_NAME" --cluster "$CLUSTER" --resource-spec "$SPEC" --image "$IMAGE" \
    --port reranker:8080:http --port ssh:22:tcp --ssh-key "$SSHKEY" \
    --init-script "$INIT_SCRIPT" -o json)"
  SLUG="$(printf '%s' "$CREATE" | jq -r '(.slug // .name) // empty')"
  [ -n "$SLUG" ] || SLUG="$WS_NAME"
fi

# ── 2) running + 노출 URL 대기 ───────────────────────────────────────────
echo "[2/3] 워크스페이스 준비 + URL 대기 (slug=$SLUG)"
URL="${RERANKER_URL_OVERRIDE:-}"
if [ -z "$URL" ]; then
  for i in $(seq 1 60); do
    JSON="$(vesslctl workspace show "$SLUG" -o json 2>/dev/null || true)"
    URL="$(printf '%s' "$JSON" | extract_url)"
    [ -n "$URL" ] && break
    printf '  ... (%d/60) 준비 중\n' "$i"; sleep 10
  done
fi
if [ -z "$URL" ]; then
  echo "URL을 못 찾음. 'vesslctl workspace show $SLUG -o json' 원본:" >&2
  printf '%s\n' "${JSON:-}" >&2
  echo "→ 위 JSON에서 8080 https 엔드포인트 확인 후 RERANKER_URL_OVERRIDE=<url> ./up.sh 로 재실행" >&2
  exit 1
fi
echo "  URL: $URL"

# ── 3) /health/ready + Redis 등록 ───────────────────────────────────────
echo "[3/3] /health/ready 대기 (GPU/모델 로딩)"
OK=""
for i in $(seq 1 30); do
  if curl -fsS --max-time 5 "${URL%/}/health/ready" >/dev/null 2>&1; then OK=1; break; fi
  printf '  ... (%d/30) not ready\n' "$i"; sleep 10
done
[ -n "$OK" ] || { echo "health/ready 실패 — 모델 로딩 지연 또는 URL 오류. 로그: vesslctl workspace logs $SLUG" >&2; exit 1; }

"$DIR/set-url.sh" "$URL"
echo
echo "✓ GPU 리랭커 ON. 사용 끝나면: ./down.sh  (크레딧 절약)"
