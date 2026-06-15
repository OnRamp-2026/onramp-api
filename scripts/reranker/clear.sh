#!/usr/bin/env bash
# 리랭커 URL을 Redis에서 삭제 → onramp-api는 URL 없음 → 서킷브레이커/vector 폴백.
# GPU(VESSL)를 내릴 때 호출. 사용: ./clear.sh
#
# 환경변수: RERANKER_REDIS_NS=onramp  RERANKER_REDIS_WORKLOAD=deploy/redis  RERANKER_REDIS_KEY=reranker:service_url
set -euo pipefail

NS="${RERANKER_REDIS_NS:-onramp}"
WORKLOAD="${RERANKER_REDIS_WORKLOAD:-deploy/redis}"
KEY="${RERANKER_REDIS_KEY:-reranker:service_url}"

kubectl -n "$NS" exec "$WORKLOAD" -- redis-cli DEL "$KEY" >/dev/null
echo "✓ Redis DEL $KEY  (ns=$NS, $WORKLOAD)"
echo "  → onramp-api 자동 vector 폴백 (GPU 없어도 서비스 정상)"
