#!/usr/bin/env bash
# on-demand GPU 리랭커 켜기: VESSL Workspace(L40S) 생성 → uvicorn 자동기동(init-script)
# → 노출 URL 구성 → /health/ready 확인 → Redis에 등록(set-url.sh).
# 끝나면 반드시 ./down.sh 로 내려서 크레딧 절약($1.80/hr).
#
# 사용: ./up.sh
# 환경변수(override 가능):
#   VESSL_WS_NAME=onramp-reranker
#   VESSL_CLUSTER=cluster-betelgeuse
#   VESSL_SPEC=resourcespec-eaac8rxxpz6j      # L40S ×1 ($1.80/hr)
#   VESSL_IMAGE=ghcr.io/onramp-2026/onramp-reranker:gpu-v1
#   VESSL_SSHKEY=sshkey-umdldri4c2ha
#   RERANKER_URL_OVERRIDE=...                 # URL 구성 실패 시 직접 지정
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

# 워크스페이스 객체(JSON) 조회 — list 구조는 [{billingInfo, workspace:{...}}].
# 같은 이름의 terminated/실패 워크스페이스가 남아 죽은 슬러그를 잡지 않게:
# 살아있는(terminated/failed 아닌) 것 중 최신(createdDt) 하나만 반환. 없으면 빈 출력(→새로 생성).
ws_json() {
  vesslctl workspace list -o json 2>/dev/null | jq -c --arg n "$WS_NAME" '
    map(.workspace)
    | map(select(.name==$n))
    | map(select((.state // "" | ascii_downcase | test("terminat|fail|error")) | not))
    | sort_by(.createdDt) | last // empty'
}
# 노출 URL 구성 — show/list에 URL 필드가 없어 규칙으로 생성(UI와 동일):
#   https://<portName>-<slug>.<region>.cloud.vessl.ai   (region = clusterSlug의 'cluster-' 제거)
ws_url() {
  jq -r '(.clusterSlug | sub("^cluster-"; "")) as $sub | (.ports[0].name) as $p
         | "https://\($p)-\(.slug).\($sub).cloud.vessl.ai"' 2>/dev/null
}

# ── 1) 워크스페이스 생성(또는 실행 중이면 재사용) ─────────────────────────
WS="$(ws_json)"
if [ -n "$WS" ]; then
  echo "[1/3] 기존 워크스페이스 재사용: $(printf '%s' "$WS" | jq -r '.slug')"
else
  echo "[1/3] 워크스페이스 생성: $WS_NAME (L40S, $IMAGE)"
  # 포트는 8080(reranker)만 — 22(ssh)/8888(jupyter)은 VESSL 예약이라 지정 불가. SSH는 VESSL이 자체 제공.
  vesslctl workspace create \
    --name "$WS_NAME" --cluster "$CLUSTER" --resource-spec "$SPEC" --image "$IMAGE" \
    --port reranker:8080:http --ssh-key "$SSHKEY" \
    --init-script "$INIT_SCRIPT" -o json >/dev/null
fi

# ── 2) running 대기 + URL 구성 ───────────────────────────────────────────
echo "[2/3] running 대기 + URL 구성"
URL="${RERANKER_URL_OVERRIDE:-}"
if [ -z "$URL" ]; then
  for i in $(seq 1 60); do
    WS="$(ws_json)"
    if [ "$(printf '%s' "$WS" | jq -r '.state // empty')" = "running" ]; then
      URL="$(printf '%s' "$WS" | ws_url)"
      [ -n "$URL" ] && break
    fi
    printf '  ... (%d/60) 준비 중\n' "$i"; sleep 10
  done
fi
if [ -z "$URL" ]; then
  echo "URL 구성 실패. 'vesslctl workspace list -o json' 확인:" >&2
  vesslctl workspace list -o json >&2
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
[ -n "$OK" ] || { echo "health/ready 실패 — 모델 로딩 지연 또는 URL 오류. 로그: vesslctl workspace logs $(printf '%s' "$WS" | jq -r '.slug')" >&2; exit 1; }

"$DIR/set-url.sh" "$URL"
echo
echo "✓ GPU 리랭커 ON. 사용 끝나면: ./down.sh  (크레딧 절약)"
