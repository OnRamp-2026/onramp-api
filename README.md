# OnRamp API

> 자연어 질문에 구조화된 답변을 생성하는 RAG 백엔드 서버 (FastAPI + LangGraph)

Confluence에 축적된 사내 지식을 자연어로 검색하고, 5요소(현재상황·원인·근거·해결·인프라) 구조화 답변을 생성합니다.

---

## Architecture

    User → FastAPI → LangGraph Workflow
                          │
                          ├── Router Agent      (질문 분류, 도메인 라우팅)
                          ├── Retriever Agent   (Qdrant Dense Search + Reranker)
                          ├── Answer Agent      (5요소 답변 생성)
                          └── Trust Agent [P1]  (5축 신뢰도 평가)

## Tech Stack

| 영역 | 기술 |
|---|---|
| Framework | FastAPI, LangGraph |
| LLM | gpt-4o-mini, GPT-4o, Azure (Sovereign 선택) |
| Embedding | text-embedding-3-small |
| Reranker | bge-reranker-v2-m3 |
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
  make install          의존성 + pre-commit 설치
  make clean            캐시 파일 정리

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

    # Confluence
    CONFLUENCE_BASE_URL=https://your-domain.atlassian.net
    CONFLUENCE_API_TOKEN=...

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

## Related Repositories

| 레포 | 설명 |
|---|---|
| infra | EKS · Terraform · Helm · 모니터링 · CI/CD |
| confluence-data-crawler | Confluence 일배치 수집 · 청킹 · 임베딩 · Qdrant 적재 |
| onramp-web | Vue 3 챗봇 프론트엔드 |
