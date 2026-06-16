# 데이터 적재

Confluence·GitHub 원문을 같은 RAG 파이프라인으로 정제·청킹·임베딩해 PostgreSQL·Qdrant·OpenSearch에 반영한다. 적재 코드는 환경을 모르고 앱 env(`DATABASE_URL`/`QDRANT_*`/`OPENSEARCH_*`/`*_TOKEN`)만 보므로 **로컬과 prod에서 같은 스크립트가 동작**한다.

## 적재 대상

| 저장소 | 역할 | 생성 조건 |
|---|---|---|
| PostgreSQL `source_document` | 원문 원장(raw + cleaned markdown), `source=confluence\|github` 구분 | 항상 |
| PostgreSQL `chunk_registry` | 청크 hash·색인 상태 추적, 재실행 시 dedup 기준 | 항상 |
| Qdrant `onramp` | dense vector 검색용 child chunk | 항상 |
| OpenSearch `onramp-chunks` | BM25 청크 검색용 인덱스 | `BM25_SEARCH_ENABLED=true` |
| OpenSearch `onramp-documents` | 문서 단위 조회/검색용 인덱스 | `index_documents_to_opensearch.py` 실행 시 |

## 스크립트

| 스크립트 | 용도 | 설명 |
|---|---|---|
| `scripts/ingest_all.sh` | 공용 코어 | migrate → Confluence 전체 적재 → GitHub 적재 → OpenSearch 문서 투영 → 현황. 로컬·prod 공용 |
| `scripts/ingest_all_local.sh` | 로컬 전체 적재 | `docker compose up`으로 인프라를 띄운 뒤 `ingest_all.sh` 호출 |
| `scripts/index_recent_confluence_pages.py` | Confluence 단독 적재 | `--all` 전체 또는 `--hours N` 증분 |
| `scripts/index_github.py` | GitHub 단독 적재 | repo 문서·이슈·PR 적재(전체 fetch + hash dedup) |
| `scripts/index_documents_to_opensearch.py` | OpenSearch 문서 투영 | PostgreSQL 원장을 `onramp-documents` 인덱스로 투영 |
| `scripts/ingest_status.py` | 적재 현황 확인 | Postgres/Qdrant/OpenSearch 카운트(docker/kubectl 비의존) |
| `scripts/fetch_recent_confluence_pages.py` | 정제 결과 확인 | Confluence HTML → Markdown만. DB/Qdrant 불필요 |
| `scripts/prepare_recent_confluence_pages.py` | 청크 JSONL 확인 | 마스킹·청킹·분류 결과를 파일로. 색인은 안 함 |

공통 env: `CONFLUENCE_LIMIT`(기본 500) · `GITHUB_REPOS`(기본 org 전체) · `SKIP_MIGRATE`/`SKIP_CONFLUENCE`/`SKIP_GITHUB`/`SKIP_DOCS=1`.

## 로컬 적재

`.env`에 최소 `OPENAI_API_KEY`, `CONFLUENCE_*`를 넣고 실행한다. GitHub·OpenSearch는 설정이 있을 때만 동작한다.

    bash scripts/ingest_all_local.sh

    # 소량 테스트
    CONFLUENCE_LIMIT=20 SKIP_GITHUB=1 bash scripts/ingest_all_local.sh

    # GitHub 일부 repo만
    GITHUB_REPOS="onramp-api docs" bash scripts/ingest_all_local.sh

    # 현황만
    python scripts/ingest_status.py

단계별 로컬 절차는 [`local_ingestion.md`](local_ingestion.md) 참고.

## Production 적재

인프라가 이미 클러스터에 있으므로 로컬 래퍼 대신 `onramp-api` 파드 안에서 공용 코어만 실행한다.

    NS=onramp
    kubectl exec -n "$NS" deploy/onramp-api -- bash scripts/ingest_all.sh
    kubectl exec -n "$NS" deploy/onramp-api -- python scripts/ingest_status.py

    # 일부 단계 제외
    kubectl exec -n "$NS" deploy/onramp-api -- env SKIP_GITHUB=1 CONFLUENCE_LIMIT=1000 bash scripts/ingest_all.sh

주의:

- 네임스페이스의 `onramp-api` Secret/ConfigMap에 `DATABASE_URL`, `QDRANT_*`, `OPENAI_API_KEY`, `CONFLUENCE_*`가 주입되어 있어야 한다.
- GitHub 적재가 필요하면 `GITHUB_TOKEN`을 Secret으로 추가한다(없으면 GitHub 단계 자동 스킵).
- OpenSearch가 없거나 `BM25_SEARCH_ENABLED=false`면 BM25 단계는 비치명적으로 스킵되고 Qdrant/Postgres만 적재된다.
- 멀티테넌트는 네임스페이스별 저장소가 다르므로 테넌트마다 따로 실행한다.
- 정기 동기화 CronJob은 아직 없다(수동 실행).

## 전체 적재와 증분 동기화

모든 소스는 색인 단계에서 hash dedup을 수행한다. 같은 `cleaned_markdown`은 재임베딩·재색인을 건너뛰므로 `ingest_all.sh` 재실행은 idempotent하다.

| 소스 | 전체 적재 | 시간 증분 fetch | 재실행 |
|---|---|---|---|
| Confluence | `index_recent_confluence_pages.py --all` | `--hours N` | 변경 없는 문서는 hash로 스킵 |
| GitHub | `index_github.py` 기본 | 현재 없음 | 전체 fetch 후 변경 없는 문서는 hash로 스킵 |

    # 초기 구축·재구축
    bash scripts/ingest_all.sh

    # Confluence만 가볍게 증분 동기화
    python scripts/index_recent_confluence_pages.py --hours 24 --limit 500

## Confluence 단독 디버깅

수집·정제만 확인할 때는 DB/Qdrant가 없어도 된다.

    # HTML → Markdown 정제만
    python scripts/fetch_recent_confluence_pages.py --hours 24 --limit 50 --output-dir data/cleaned/recent --save-html
    python scripts/fetch_recent_confluence_pages.py --all --limit 500 --output-dir data/cleaned/all

    # 청크 JSONL만
    python scripts/prepare_recent_confluence_pages.py --hours 24 --limit 50 --output-dir data/processed/prepared_chunks

Confluence 페이지 수정 테스트는 기본 dry-run이며 `--apply`로 실제 적용한다.

    python scripts/random_confluence_page_editor.py --count 3 --candidate-limit 100          # dry-run
    python scripts/random_confluence_page_editor.py --count 3 --candidate-limit 100 --apply  # 실제 수정
