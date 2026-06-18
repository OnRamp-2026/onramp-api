#!/usr/bin/env bash
# 전체 적재 코어 (로컬·prod 공용, 인프라 비의존).
#   마이그레이션 → Confluence → GitHub → OpenSearch 문서 투영 → 현황
#
# 저장소 주소는 전부 앱 env(DATABASE_URL / QDRANT_* / OPENSEARCH_* / *_TOKEN)에서 읽는다.
# 따라서 docker나 kubectl을 모른다 — 인프라가 떠 있는 곳이면 어디서든 동일하게 동작:
#   - 로컬:  scripts/ingest_all_local.sh 가 docker compose 띄운 뒤 이 스크립트를 호출
#   - prod: onramp-api 파드 안에서 직접 실행 (kubectl exec / k8s Job) — README 참고
#
# env 조절:
#   CONFLUENCE_LIMIT(기본 500) / GITHUB_REPOS(기본 org 전체)
#   SKIP_MIGRATE / SKIP_CONFLUENCE / SKIP_GITHUB / SKIP_DOCS = 1 로 단계 스킵
set -uo pipefail
cd "$(dirname "$0")/.."

CONFLUENCE_LIMIT="${CONFLUENCE_LIMIT:-5000}"
GITHUB_REPOS="${GITHUB_REPOS:-onramp-api docs onramp-web onramp-reranker onramp-stt-api gitops infra monitoring stt_correction confluence-data-crawler}"
SKIP_MIGRATE="${SKIP_MIGRATE:-0}"
SKIP_CONFLUENCE="${SKIP_CONFLUENCE:-0}"
SKIP_GITHUB="${SKIP_GITHUB:-0}"
SKIP_DOCS="${SKIP_DOCS:-0}"
# REINDEX=1 → content-hash dedup 무시하고 전체 재색인(도메인 분류만 바꿔 재적재 시). 전체 wipe 불필요.
REINDEX="${REINDEX:-0}"
REINDEX_FLAG=""
[ "$REINDEX" = "1" ] && REINDEX_FLAG="--reindex"

log() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
warn() { printf '\033[1;33m! %s\033[0m\n' "$*"; }

# venv 자동 활성화(로컬). prod 파드는 시스템 파이썬이라 .venv 없음 → 그대로 진행.
if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# 1) 마이그레이션
if [ "$SKIP_MIGRATE" != "1" ]; then
  log "DB 마이그레이션 (alembic upgrade head)"
  alembic upgrade head || { warn "마이그레이션 실패 — 중단"; exit 1; }
fi

# 2) Confluence
if [ "$SKIP_CONFLUENCE" != "1" ]; then
  log "Confluence 적재 (limit=$CONFLUENCE_LIMIT)"
  # shellcheck disable=SC2086
  python scripts/index_recent_confluence_pages.py --all --limit "$CONFLUENCE_LIMIT" $REINDEX_FLAG || warn "Confluence 적재 실패 — 계속"
fi

# 3) GitHub (유효 토큰 있을 때만)
if [ "$SKIP_GITHUB" != "1" ]; then
  if [ -n "${GITHUB_TOKEN:-}" ] || grep -qE '^GITHUB_TOKEN=("?)(ghp_|github_pat_)' .env 2>/dev/null; then
    log "GitHub 적재 (repos: $GITHUB_REPOS)"
    # shellcheck disable=SC2086
    python scripts/index_github.py --repos $GITHUB_REPOS $REINDEX_FLAG || warn "GitHub 적재 실패 — 계속"
  else
    warn "GitHub 스킵 — GITHUB_TOKEN 미설정(env 또는 .env)"
  fi
fi

# 4) OpenSearch 문서 인덱스 투영 (Postgres 원문 → onramp-documents)
if [ "$SKIP_DOCS" != "1" ]; then
  log "OpenSearch 문서 인덱스 투영"
  python scripts/index_documents_to_opensearch.py || warn "문서 투영 실패(OpenSearch 미기동?) — 계속"
fi

# 5) 현황 (저장소 직접 조회 — docker/kubectl 비의존)
log "적재 현황"
python scripts/ingest_status.py || true

log "완료"
