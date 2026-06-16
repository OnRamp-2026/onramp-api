# OnRamp API

> 자연어 질문에 구조화된 답변을 생성하는 RAG 백엔드 서버 (FastAPI + LangGraph)

Confluence에 축적된 사내 지식을 자연어로 검색하고, 5요소(현재상황·원인·근거·해결·인프라) 구조화 답변을 생성합니다.

---

## Architecture

    User → FastAPI → LangGraph Workflow
                          │
                          ├── Router Agent      (질문 분류, 도메인 라우팅 / 범위 밖 질문 차단)
                          ├── Retriever Agent   (Qdrant Dense Search + Reranker)
                          ├── Trust Agent       (Evidence Confidence 5축 채점 → 근거 부족 시 재검색)
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

    # (선택) 리랭커 의존성 — 미설치 시 vector score 순 폴백으로 동작
    make install-rerank          # backend 선택은 아래 '리랭커' 섹션 참고

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

Confluence·GitHub 원문을 PostgreSQL(원장)·Qdrant(dense)·OpenSearch(BM25)에 적재한다. 같은 스크립트가 로컬·prod에서 동작한다(차이는 env뿐).

    # 로컬 — 인프라 기동 + 전체 적재 + 현황
    bash scripts/ingest_all_local.sh

    # Production — 파드 안에서 공용 코어 실행
    kubectl exec -n onramp deploy/onramp-api -- bash scripts/ingest_all.sh

    # 현황 확인 (어디서든)
    python scripts/ingest_status.py

스크립트별 역할, 로컬·prod 절차, 증분/전체 동기화, Confluence 단독 디버깅은 [`docs/ingestion.md`](docs/ingestion.md) 참고.

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

`RERANKER_BACKEND=remote`면 리랭킹을 별도 서비스(`onramp-reranker`)에 위임한다(메모리 격리, `RERANKER_SERVICE_URL`). 실패 시 vector 폴백. 별도 GPU 서비스 운영(on-demand 포함)은 [`scripts/reranker/README.md`](scripts/reranker/README.md) 참고.

## Documentation

| 문서 | 내용 |
|---|---|
| [`docs/ingestion.md`](docs/ingestion.md) | 데이터 적재 — 저장소·스크립트·로컬/prod·증분 |
| [`docs/local_ingestion.md`](docs/local_ingestion.md) | 로컬 적재 단계별 절차 |
| [`scripts/reranker/README.md`](scripts/reranker/README.md) | (선택) remote 리랭커 운영 스크립트 |

## Related Repositories

| 레포 | 설명 |
|---|---|
| infra | EKS · Terraform · Helm · 모니터링 · CI/CD |
| confluence-data-crawler | Confluence 일배치 수집 · 청킹 · 임베딩 · Qdrant 적재 |
| onramp-web | Vue 3 챗봇 프론트엔드 |
