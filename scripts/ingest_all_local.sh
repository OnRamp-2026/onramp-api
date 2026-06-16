#!/usr/bin/env bash
# 로컬 전체 적재 한 방 스크립트 (골든셋 구축용).
#   인프라 기동 → 마이그레이션 → Confluence → GitHub → OS 문서 투영 → 검증
#
# 사용:
#   bash scripts/ingest_all_local.sh
#   CONFLUENCE_LIMIT=20 SKIP_GITHUB=1 bash scripts/ingest_all_local.sh   # 일부만
#   GITHUB_REPOS="onramp-api docs" bash scripts/ingest_all_local.sh
#
# env 조절:
#   CONFLUENCE_LIMIT(기본 500) / GITHUB_REPOS(기본 org 전체)
#   SKIP_INFRA / SKIP_CONFLUENCE / SKIP_GITHUB / SKIP_DOCS = 1 로 단계 스킵
set -uo pipefail
cd "$(dirname "$0")/.."

CONFLUENCE_LIMIT="${CONFLUENCE_LIMIT:-500}"
GITHUB_REPOS="${GITHUB_REPOS:-onramp-api docs onramp-web onramp-reranker onramp-stt-api gitops infra monitoring stt_correction confluence-data-crawler}"
SKIP_INFRA="${SKIP_INFRA:-0}"
SKIP_CONFLUENCE="${SKIP_CONFLUENCE:-0}"
SKIP_GITHUB="${SKIP_GITHUB:-0}"
SKIP_DOCS="${SKIP_DOCS:-0}"

log() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
warn() { printf '\033[1;33m! %s\033[0m\n' "$*"; }

# venv 자동 활성화(있으면)
if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# 1) 인프라 (idempotent — 이미 떠 있으면 그대로)
if [ "$SKIP_INFRA" != "1" ]; then
  log "인프라 기동 (docker compose up -d)"
  docker compose up -d
  log "postgres 준비 대기"
  until docker compose exec -T postgres pg_isready -U postgres >/dev/null 2>&1; do sleep 1; done
  log "opensearch 준비 대기 (없으면 BM25/문서 단계는 스킵됨)"
  for _ in $(seq 1 30); do curl -fsS localhost:9200/_cluster/health >/dev/null 2>&1 && break; sleep 2; done
fi

# 2) 마이그레이션 (실패 시 중단)
log "DB 마이그레이션 (alembic upgrade head)"
alembic upgrade head || { warn "마이그레이션 실패 — 중단"; exit 1; }

# 3) Confluence (실패해도 다음 단계 진행)
if [ "$SKIP_CONFLUENCE" != "1" ]; then
  log "Confluence 적재 (limit=$CONFLUENCE_LIMIT)"
  python scripts/index_recent_confluence_pages.py --all --limit "$CONFLUENCE_LIMIT" || warn "Confluence 적재 실패 — 계속"
fi

# 4) GitHub (유효 토큰 있을 때만)
if [ "$SKIP_GITHUB" != "1" ]; then
  if grep -qE '^GITHUB_TOKEN=("?)(ghp_|github_pat_)' .env 2>/dev/null; then
    log "GitHub 적재 (repos: $GITHUB_REPOS)"
    # shellcheck disable=SC2086
    python scripts/index_github.py --repos $GITHUB_REPOS || warn "GitHub 적재 실패 — 계속"
  else
    warn "GitHub 스킵 — .env에 유효한 GITHUB_TOKEN(ghp_/github_pat_) 없음"
  fi
fi

# 5) OpenSearch 문서 인덱스 투영 (Postgres 원문 → onramp-documents)
if [ "$SKIP_DOCS" != "1" ]; then
  log "OpenSearch 문서 인덱스 투영"
  python scripts/index_documents_to_opensearch.py || warn "문서 투영 실패(OpenSearch 미기동?) — 계속"
fi

# 6) 검증
log "적재 확인"
docker compose exec -T postgres psql -U postgres -d onramp -c \
  "SELECT source, count(*) AS docs, max(length(cleaned_markdown)) AS max_len FROM source_document GROUP BY source;" || true
docker compose exec -T postgres psql -U postgres -d onramp -tc \
  "SELECT 'chunk_registry rows: ' || count(*) FROM chunk_registry;" || true
curl -s localhost:6333/collections/onramp 2>/dev/null \
  | python -c "import sys,json;print('qdrant points:',json.load(sys.stdin)['result']['points_count'])" 2>/dev/null || true
printf 'opensearch chunks: ';  curl -s localhost:9200/onramp-chunks/_count 2>/dev/null;    echo
printf 'opensearch documents: '; curl -s localhost:9200/onramp-documents/_count 2>/dev/null; echo

log "완료"
