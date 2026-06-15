#!/usr/bin/env bash
# 리랭커(VESSL) URL을 클러스터 Redis에 등록 → onramp-api가 런타임 조회(rollout 없이 반영, ~30s).
# 사용: ./set-url.sh https://reranker-wsp-xxxx.betelgeuse.cloud.vessl.ai
#
# 환경변수(기본값 override):
#   RERANKER_REDIS_NS=onramp  RERANKER_REDIS_WORKLOAD=deploy/redis  RERANKER_REDIS_KEY=reranker:service_url
set -euo pipefail

URL="${1:-}"
[ -n "$URL" ] || { echo "사용법: $0 <reranker-url>" >&2; exit 1; }
# http/https + host 형식만 허용(앱의 _is_http_url 검증과 동일 기준 — 오염 값 방지).
case "$URL" in
  http://*|https://*) : ;;
  *) echo "URL은 http:// 또는 https:// 로 시작해야 합니다: $URL" >&2; exit 1 ;;
esac

NS="${RERANKER_REDIS_NS:-onramp}"
WORKLOAD="${RERANKER_REDIS_WORKLOAD:-deploy/redis}"
KEY="${RERANKER_REDIS_KEY:-reranker:service_url}"

kubectl -n "$NS" exec "$WORKLOAD" -- redis-cli SET "$KEY" "$URL" >/dev/null
echo "✓ Redis SET $KEY = $URL  (ns=$NS, $WORKLOAD)"
echo "  → onramp-api가 다음 조회(최대 ~30s)부터 이 URL로 리랭킹"
