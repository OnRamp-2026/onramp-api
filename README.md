# OnRamp API

> 자연어 질문에 구조화된 답변을 생성하는 RAG 백엔드 서버 (FastAPI + LangGraph)

Confluence에 축적된 사내 지식을 자연어로 검색하고, 5요소(현재상황·원인·근거·해결·인프라) 구조화 답변을 생성합니다.

---

## Architecture

    User → FastAPI → LangGraph Workflow
                          │
                          ├── Router Agent      (질문 분류, 도메인 라우팅 / 범위 밖 질문 차단)
                          ├── Retriever Agent   (Qdrant Dense Search + Reranker)
                          ├── Trust Agent [P1]  (Evidence Confidence 5축 채점 → 근거 부족 시 재검색)
                          └── Answer Agent      (Answerability Status 판단 → 5요소 답변 생성/보류)

    실행 순서: Router → Retriever → Trust → (근거 부족 시 Retriever 재검색) → Answer

## Tech Stack

| 영역 | 기술 |
|---|---|
| Framework | FastAPI, LangGraph |
| LLM | gpt-4o-mini, GPT-4o, Azure (Sovereign 선택) |
| Embedding | text-embedding-3-small |
| Reranker | bge-reranker-v2-m3 — 환경별 backend (torch CPU/GPU · ONNX int8 CPU · remote 서비스) |
| Vector DB | Qdrant |
| DB | PostgreSQL (asyncpg + SQLAlchemy) |
| Cache | Redis |
| Infra | EKS (별도 infra 레포 관리) |

## Project Structure

    app/
    ├── api/           # 엔드포인트 (v1/chat, v1/asset, v1/health)
    ├── agents/        # LangGraph Agent (router, retriever, answer, trust)
    ├── rag/           # RAG 코어 (embedder, chunker, classifier, reranker)
    ├── services/      # 비즈니스 로직 (chat_service, asset_service)
    ├── db/            # 데이터 접근 (qdrant, postgres, redis, confluence)
    ├── middleware/     # Request ID, 로깅, 에러 핸들링
    └── models/        # Pydantic 스키마 (request, response, domain)

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (패키지 매니저)
- Docker & Docker Compose (로컬 개발)

### Setup

    # 레포 클론
    git clone https://github.com/your-org/onramp-api.git
    cd onramp-api

    # uv 설치 (없는 경우)
    pip install uv

    # 가상환경 생성 + 활성화
    uv venv .venv
    source .venv/bin/activate        # Mac/Linux
    # .venv\Scripts\activate         # Windows

    # 의존성 설치 (프로덕션 + 개발)
    uv pip install -e ".[dev]"

    # (선택) 리랭커 의존성 — CPU torch 휠 고정 설치 (CUDA 미포함, 이미지/디스크 경량화)
    # 미설치 시 retriever는 vector score 순 폴백으로 정상 동작한다.
    make install-rerank

    # (선택) ONNX(int8) 리랭커 백엔드 — CPU 추론 경량화 (#60, opt-in · 기본 backend는 torch)
    #   ① 셋업(최초 1회)  make setup-reranker-onnx     # = install-onnx + build-reranker-onnx
    #   ② 활성화          .env에 RERANKER_BACKEND=onnx, RERANKER_ONNX_DIR=models/bge-reranker-onnx-int8
    #   (운영 x86 파드는 ARCH=avx512_vnni 로 재생성: make build-reranker-onnx ARCH=avx512_vnni)
    #   (속도/품질 비교: make bench-reranker-onnx)

    # 환경변수 설정
    cp .env.example .env
    # .env 파일에 API 키 등 입력

    # 로컬 인프라 실행 (Qdrant, PostgreSQL, Redis)
    docker compose up -d

    # DB 마이그레이션
    alembic upgrade head

    # 서버 실행
    make dev

### Makefile Commands
```
  make dev              개발 서버 실행 (--reload)      
  make test             전체 테스트     
  make test-unit        단위 테스트만      
  make test-cov         커버리지 리포트 생성    
  make lint             린트 검사   
  make format           자동 포맷 + 린트 수정   
  make typecheck        mypy 타입 체크    
  make migrate          DB 마이그레이션 적용   
  make migrate-new      새 마이그레이션 생성   
  make up               로컬 인프라 실행 (docker compose)   
  make down             로컬 인프라 중지    
  make install          의존성 + pre-commit 설치 (1회성)   
  make install-rerank   리랭커 의존성(CPU torch + sentence-transformers) (1회성)   
  make setup-reranker-onnx  ONNX 리랭커 셋업 = install-onnx + build (최초 1회)   
  make install-onnx       ↳ ONNX 의존성만 설치 (1회성)   
  make build-reranker-onnx  ↳ ONNX int8 산출물 (ARCH=arm64|avx512_vnni)   
  make bench-reranker-onnx  torch vs ONNX(int8) 속도·품질 벤치   
  make clean            캐시 파일 정리
```

## API Endpoints

### POST /v1/chat

자연어 질문 → 5요소 구조화 답변

Request:

    {
      "query": "EKS Pod CrashLoopBackOff 어떻게 해결해?",
      "model": "gpt-4o-mini"
    }

Response:

    {
      "answer": {
        "situation": "...",
        "cause": "...",
        "evidence": "...",
        "solution": "...",
        "infra_context": "..."
      },
      "sources": [...],
      "trust_score": { ... }
    }

### POST /v1/asset

회의 녹취 텍스트 → 5요소 보고서 생성 → Confluence 등록

    {
      "transcript": "회의 녹취 텍스트...",
      "category": "장애대응"
    }

### STT 자동 보고서 worker

업로드 workflow 이후 STT 이벤트 소비와 보고서 생성을 별도 프로세스로 실행한다.

    python -m app.workers.outbox_publisher
    python -m app.workers.stt_event_consumer
    python -m app.workers.report_generator

생성된 보고서는 `GET/PATCH /v1/reports/{report_id}`와
`POST /v1/reports/{report_id}/approve`에서 조회, 수정, 승인한다.
긴 전사문은 `REPORT_WINDOW_MAX_CHARS`와 `REPORT_WINDOW_OVERLAP_CHARS` 기준으로
구간별 추출 후 최종 보고서로 병합한다.

### GET /v1/health

서비스 상태 확인

## Environment Variables

    # LLM
    OPENAI_API_KEY=sk-...
    AZURE_OPENAI_ENDPOINT=https://...
    AZURE_OPENAI_API_KEY=...

    # Vector DB
    QDRANT_HOST=localhost
    QDRANT_PORT=6333

    # Database
    DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/onramp

    # Cache
    REDIS_URL=redis://localhost:6379/0

    # Reranker — 환경에 따라 backend 선택 (어느 backend든 실패 시 vector score 폴백)
    RERANKER_BACKEND=torch          # torch | onnx | remote
    RERANKER_DEVICE=cpu             # torch 전용: cpu(로컬·CPU 노드) | cuda(GPU 노드)
    # ONNX int8 — GPU 없는 노드/로컬 CPU 경량 추론 (make setup-reranker-onnx):
    # RERANKER_BACKEND=onnx
    # RERANKER_ONNX_DIR=models/bge-reranker-onnx-int8
    # remote — 리랭커를 별도 서비스로 분리(메모리 격리, 선택):
    # RERANKER_BACKEND=remote
    # RERANKER_SERVICE_URL=http://onramp-reranker:8080

    # Confluence
    CONFLUENCE_BASE_URL=https://your-domain.atlassian.net
    CONFLUENCE_API_TOKEN=...
    CONFLUENCE_USER_EMAIL=your@email.com
    CONFLUENCE_SPACE_KEY=OnRamp
    CONFLUENCE_TIMEZONE=Asia/Seoul

## 리랭커 (환경별 backend)

검색 후보를 cross-encoder(`bge-reranker-v2-m3`)로 재정렬한다. **배포 환경에 맞춰 backend를 고른다** — 같은 이미지·코드에서 env(`RERANKER_BACKEND`)로만 전환한다. **어느 backend든 실패하면 vector score 순으로 폴백**하므로 리랭커가 없거나 죽어도 API는 정상 동작한다.

| 환경 | backend | 설정 |
|---|---|---|
| GPU 노드 (prod) | `torch` (GPU) | `RERANKER_BACKEND=torch` · `RERANKER_DEVICE=cuda` |
| CPU 노드 / 로컬 | `onnx` (int8 경량) | `RERANKER_BACKEND=onnx` · `RERANKER_ONNX_DIR=...` (`make setup-reranker-onnx`) |
| CPU, 의존성 최소 | `torch` (CPU) | `RERANKER_BACKEND=torch` · `RERANKER_DEVICE=cpu` |
| (선택) 분리 운영 | `remote` | 리랭커를 별도 서비스(`onramp-reranker`)로 — 메모리 격리. `RERANKER_SERVICE_URL=...` |

일반적으로 **GPU 노드면 `torch`+`cuda`**, **GPU가 없으면 `onnx` int8**(CPU 추론 경량화)로 간다. backend는 환경별 설정(`values-*.yaml` / `.env`)으로 결정한다.

## 데이터 적재

권장 적재 경로는 **멀티소스 적재 스크립트**입니다. Confluence 개별 스크립트는 정제 결과 확인이나 부분 디버깅용으로만 사용합니다.

### 적재 대상

Confluence와 GitHub 원문을 같은 RAG 파이프라인으로 정제·청킹·임베딩하고, 아래 저장소에 반영합니다.

| 저장소 | 역할 | 생성 조건 |
|---|---|---|
| PostgreSQL `source_document` | 원문 원장(raw + cleaned markdown), `source=confluence\|github` 구분 | 항상 |
| PostgreSQL `chunk_registry` | 청크 hash·색인 상태 추적, 재실행 시 dedup 기준 | 항상 |
| Qdrant `onramp` | dense vector 검색용 child chunk | 항상 |
| OpenSearch `onramp-chunks` | BM25 청크 검색용 인덱스 | `BM25_SEARCH_ENABLED=true` |
| OpenSearch `onramp-documents` | 문서 단위 조회/검색용 인덱스 | `index_documents_to_opensearch.py` 실행 시 |

### 스크립트 역할

| 스크립트 | 언제 사용하나 | 설명 |
|---|---|---|
| `scripts/ingest_all.sh` | 공용 코어 | migrate → Confluence 전체 적재 → GitHub 적재 → OpenSearch 문서 투영 → 현황 출력. 로컬·prod 모두 같은 파일을 사용 |
| `scripts/ingest_all_local.sh` | 로컬 전체 적재 | `docker compose up`으로 Postgres/Qdrant/Redis/OpenSearch를 띄운 뒤 `ingest_all.sh` 호출 |
| `scripts/index_recent_confluence_pages.py` | Confluence 단독 적재 | `--all` 전체 또는 `--hours N` 증분. 정기 Confluence 동기화나 부분 테스트에 사용 |
| `scripts/index_github.py` | GitHub 단독 적재 | repo 문서·이슈·PR 적재. 현재는 전체 fetch 후 hash dedup |
| `scripts/index_documents_to_opensearch.py` | OpenSearch 문서 투영 | PostgreSQL 원장을 `onramp-documents` 인덱스로 투영 |
| `scripts/ingest_status.py` | 적재 현황 확인 | Postgres/Qdrant/OpenSearch 카운트 출력. docker/kubectl 비의존 |
| `scripts/fetch_recent_confluence_pages.py` | 정제 결과 확인 | Confluence HTML → Markdown만 확인. DB/Qdrant 불필요 |
| `scripts/prepare_recent_confluence_pages.py` | 청크 JSONL 확인 | 마스킹·청킹·분류 결과를 파일로 확인. 색인은 하지 않음 |

### 로컬 전체 적재

`.env`에 최소한 `OPENAI_API_KEY`, `CONFLUENCE_*` 값을 넣고 실행합니다. GitHub와 OpenSearch는 설정이 있을 때만 의미가 있습니다.

    bash scripts/ingest_all_local.sh

소량 테스트:

    CONFLUENCE_LIMIT=20 SKIP_GITHUB=1 bash scripts/ingest_all_local.sh

GitHub 일부 repo만:

    GITHUB_REPOS="onramp-api docs" bash scripts/ingest_all_local.sh

현황만 확인:

    python scripts/ingest_status.py

자세한 로컬 절차는 [`docs/local_ingestion.md`](docs/local_ingestion.md)를 참고합니다.

### Production 적재

Production에서는 로컬 래퍼를 쓰지 않습니다. 인프라는 이미 클러스터에 있으므로 `onramp-api` 파드 안에서 공용 코어만 실행합니다.

    NS=onramp
    kubectl exec -n "$NS" deploy/onramp-api -- bash scripts/ingest_all.sh

현황 확인:

    kubectl exec -n "$NS" deploy/onramp-api -- python scripts/ingest_status.py

일부 단계 제외:

    kubectl exec -n "$NS" deploy/onramp-api -- env SKIP_GITHUB=1 CONFLUENCE_LIMIT=1000 bash scripts/ingest_all.sh

주의:

- 실행 대상 네임스페이스의 `onramp-api` Secret/ConfigMap에 `DATABASE_URL`, `QDRANT_*`, `OPENAI_API_KEY`, `CONFLUENCE_*`가 주입되어 있어야 합니다.
- GitHub 적재가 필요하면 `GITHUB_TOKEN`을 Secret으로 추가합니다. 없으면 GitHub 단계는 자동 스킵됩니다.
- OpenSearch가 없거나 `BM25_SEARCH_ENABLED=false`면 BM25 관련 단계는 비치명적으로 스킵되고 Qdrant/Postgres 적재만 수행됩니다.
- 멀티테넌트 환경에서는 네임스페이스별 저장소가 다르므로 테넌트마다 따로 실행합니다.
- 현재 정기 동기화 CronJob은 없습니다. 자동화는 후속 작업입니다.

### 전체 적재와 증분 동기화

모든 소스는 색인 단계에서 hash dedup을 수행합니다. 같은 `cleaned_markdown`은 재임베딩·재색인을 건너뛰므로 `ingest_all.sh` 재실행은 idempotent합니다.

| 소스 | 전체 적재 | 시간 증분 fetch | 재실행 동작 |
|---|---|---|---|
| Confluence | `index_recent_confluence_pages.py --all` | `--hours N` | 변경 없는 문서는 hash로 스킵 |
| GitHub | `index_github.py` 기본 동작 | 현재 없음 | 전체 fetch 후 변경 없는 문서는 hash로 스킵 |

권장 운영:

    # 초기 구축·재구축
    bash scripts/ingest_all.sh

    # Confluence만 가볍게 증분 동기화
    python scripts/index_recent_confluence_pages.py --hours 24 --limit 500

### Confluence 단독 디버깅

수집·정제만 확인할 때는 DB/Qdrant가 없어도 됩니다.

    python scripts/fetch_recent_confluence_pages.py --hours 24 --limit 50 --output-dir data/cleaned/recent --save-html

전체 정제 확인:

    python scripts/fetch_recent_confluence_pages.py --all --limit 500 --output-dir data/cleaned/all

청크 JSONL만 확인:

    python scripts/prepare_recent_confluence_pages.py --hours 24 --limit 50 --output-dir data/processed/prepared_chunks

Confluence 페이지 수정 테스트는 기본 dry-run입니다.

    python scripts/random_confluence_page_editor.py --count 3 --candidate-limit 100

실제로 수정하려면 `--apply`를 붙입니다.

    python scripts/random_confluence_page_editor.py --count 3 --candidate-limit 100 --apply

## Development

### 의존성 추가

    # 프로덕션 패키지 추가
    # pyproject.toml의 dependencies에 추가 후
    uv pip install -e "."

    # 개발 패키지 추가
    # pyproject.toml의 [project.optional-dependencies] dev에 추가 후
    uv pip install -e ".[dev]"

### 테스트

    # 전체
    pytest

    # 단위 테스트만
    pytest tests/unit/

    # 특정 Agent
    pytest tests/unit/test_router_agent.py -v

    # 커버리지 리포트
    pytest --cov=app --cov-report=html

### 린트 & 포맷

    # 자동 수정
    ruff check app/ --fix
    ruff format app/

    # pre-commit 설치 (push 전 자동 검사)
    pre-commit install

### DB 마이그레이션

    # 마이그레이션 파일 생성
    alembic revision --autogenerate -m "add_table_name"

    # 적용
    alembic upgrade head

    # 롤백
    alembic downgrade -1

## Deployment

EKS 배포는 infra 레포에서 Helm + ArgoCD로 관리합니다.
이 레포의 CI (Jenkinsfile)는 빌드 → 테스트 → 이미지 푸시까지 담당합니다.

    push → Jenkins → ruff + pytest → Docker build → ECR push → ArgoCD sync

### (선택) remote backend — 리랭커 분리

`RERANKER_BACKEND=remote`면 리랭킹을 별도 서비스(`onramp-reranker`)에 위임한다(메모리 격리). 일반적으로는 고정 엔드포인트(`RERANKER_SERVICE_URL=http://onramp-reranker:8080`)를 쓴다. 실패 시 vector 폴백.

> 비용 절감 실험용으로 GPU(VESSL)를 띄웠다 내리는 on-demand 스크립트(`scripts/reranker/up.sh`·`down.sh`)도 있다 — 이때만 URL이 스핀업마다 바뀌어 Redis(`reranker:service_url`)로 런타임 조회한다. **표준 운영 경로는 아니며 선택/실험 사항이다.** 자세한 내용은 [`scripts/reranker/README.md`](scripts/reranker/README.md).

## Related Repositories

| 레포 | 설명 |
|---|---|
| infra | EKS · Terraform · Helm · 모니터링 · CI/CD |
| confluence-data-crawler | Confluence 일배치 수집 · 청킹 · 임베딩 · Qdrant 적재 |
| onramp-web | Vue 3 챗봇 프론트엔드 |
