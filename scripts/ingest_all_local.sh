#!/usr/bin/env bash
# 로컬 전체 적재 (골든셋 구축용).
#   docker compose 인프라 기동 → 준비 대기 → scripts/ingest_all.sh(공용 코어) 위임
#
# prod에서는 이 래퍼가 아니라 코어(scripts/ingest_all.sh)를 파드 안에서 실행한다 (README 참고).
#
# 사용:
#   bash scripts/ingest_all_local.sh
#   CONFLUENCE_LIMIT=20 SKIP_GITHUB=1 bash scripts/ingest_all_local.sh   # 일부만
#   GITHUB_REPOS="onramp-api docs" bash scripts/ingest_all_local.sh
# (CONFLUENCE_LIMIT/GITHUB_REPOS/SKIP_* 는 코어로 그대로 전달됨)
set -uo pipefail
cd "$(dirname "$0")/.."

log() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

if [ "${SKIP_INFRA:-0}" != "1" ]; then
  log "인프라 기동 (docker compose up -d)"
  docker compose up -d
  log "postgres 준비 대기"
  until docker compose exec -T postgres pg_isready -U postgres >/dev/null 2>&1; do sleep 1; done
  log "opensearch 준비 대기 (없으면 BM25/문서 단계는 코어에서 스킵됨)"
  for _ in $(seq 1 30); do curl -fsS localhost:9200/_cluster/health >/dev/null 2>&1 && break; sleep 2; done
fi

exec bash scripts/ingest_all.sh
