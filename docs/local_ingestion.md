# 로컬 적재 런북 (골든셋 구축용)

클러스터 없이 로컬 `docker-compose`에 **Confluence·GitHub 원문을 Postgres·Qdrant·OpenSearch**에
적재하는 절차. 적재된 데이터를 보면서 골든셋(질의·정답 문서)을 만든다.

| 저장소 | 역할 | 적재 코드 |
|---|---|---|
| PostgreSQL `source_document` | 원문 진실원천(raw + cleaned_markdown), 멀티소스 원장 | IndexService / GithubIndexService |
| Qdrant `onramp-chunks` | dense 청크(벡터) | `index_children` (항상) |
| OpenSearch `onramp-chunks` | 청크 BM25(하이브리드) | `index_children` (BM25_SEARCH_ENABLED=true일 때만) |
| OpenSearch `onramp-documents` | 문서 단위 BM25(document_tools) | `index_documents_to_opensearch.py` (Postgres→투영) |

## 1. 인프라 기동

```bash
docker compose up -d          # opensearch:9200 qdrant:6333 postgres:5432 redis:6379
docker compose ps             # health 확인
```

## 2. 환경 변수 (`.env`)

```dotenv
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/onramp
QDRANT_URL=http://localhost:6333
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_SCHEME=http
BM25_SEARCH_ENABLED=true        # OpenSearch 청크 적재/하이브리드 켜기 (기본 off)
OPENAI_API_KEY=sk-...           # 임베딩 + 청크 메타 분류
# Confluence
CONFLUENCE_BASE_URL=...; CONFLUENCE_USERNAME=...; CONFLUENCE_API_TOKEN=...; CONFLUENCE_SPACE_KEY=...
# GitHub
GITHUB_TOKEN=ghp_...            # repo scope (private 포함)
GITHUB_ORG=OnRamp-2026
```

> OpenSearch 청크(하이브리드)를 안 쓸 거면 `BM25_SEARCH_ENABLED`는 빼도 된다 → Qdrant+Postgres만 적재.

## 3. DB 마이그레이션

```bash
alembic upgrade head            # source_document(_previous), chunk_registry, index_run ...
```

## 4. 적재

```bash
# (a) Confluence — 전체 또는 증분
python scripts/index_recent_confluence_pages.py --all --limit 200
#   → Qdrant 청크 + Postgres 원문(source='confluence') + (BM25 on이면) OpenSearch 청크

# (b) GitHub — repo 문서 + 이슈/PR
python scripts/index_github.py --repos onramp-api onramp-web infra gitops
#   → Qdrant 청크 + Postgres 원문(source='github') + (BM25 on이면) OpenSearch 청크
#   문서만: --no-issues / 이슈만: --no-docs

# (c) 문서 단위 BM25 — Postgres 원문 → OpenSearch onramp-documents
python scripts/index_documents_to_opensearch.py            # 전체
python scripts/index_documents_to_opensearch.py --source github
```

## 5. 적재 확인

```bash
# Postgres: 소스별 문서 수
docker compose exec postgres psql -U postgres -d onramp -c \
  "SELECT source, count(*) FROM source_document GROUP BY source;"

# Qdrant: 청크 포인트 수
curl -s localhost:6333/collections/onramp-chunks | python -m json.tool | grep points_count

# OpenSearch: 청크/문서 인덱스 카운트
curl -s 'localhost:9200/onramp-chunks/_count'
curl -s 'localhost:9200/onramp-documents/_count'
```

## 6. 골든셋 구축

적재된 `source_document`(원문)·Qdrant 청크를 보고 질의/정답 문서를 작성한다.
온보딩(팀·프로젝트 문서)·회의(meeting)·장애대응(incident) 균형을 맞춘다 →
`scripts/bootstrap_golden.py`, `scripts/pool_candidates.py`, `scripts/validate_qrels.py` 참고.

## 정리

```bash
docker compose down            # 컨테이너만 (볼륨 유지)
docker compose down -v         # 볼륨까지 삭제(데이터 초기화)
```
