# OnRamp API

> 자연어 질문에 구조화된 답변을 생성하는 사내 지식 RAG 백엔드 (FastAPI + LangGraph)

[![CI](https://github.com/OnRamp-2026/onramp-api/actions/workflows/ci.yaml/badge.svg)](https://github.com/OnRamp-2026/onramp-api/actions/workflows/ci.yaml)
![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C)
![Qdrant](https://img.shields.io/badge/Qdrant-DC244C)
![OpenSearch](https://img.shields.io/badge/OpenSearch-005EB8?logo=opensearch&logoColor=white)
![Langfuse](https://img.shields.io/badge/Langfuse-observability-FD8D3C)

Confluence·GitHub에 축적된 사내 지식을 자연어로 검색하고, 5요소(현재상황·원인·근거·해결·인프라) 구조화 답변을 생성합니다. 회의 녹취 → 보고서 자산화(STT) 파이프라인도 함께 제공합니다.

---

### Screenshots

<table>
  <tr>
    <td width="50%" align="center">
      <img src="https://placehold.co/600x360?text=Chat+Answer" alt="챗봇 답변" width="100%"><br>
      <sub>챗봇 5요소 답변 + 출처 · <code>docs/assets/chat.png</code></sub>
    </td>
    <td width="50%" align="center">
      <img src="https://placehold.co/600x360?text=Asset+Report" alt="자산화" width="100%"><br>
      <sub>회의 녹취 → 보고서 자산화 · <code>docs/assets/asset.png</code></sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="https://placehold.co/600x360?text=Langfuse+Trace" alt="Langfuse trace" width="100%"><br>
      <sub>Langfuse trace — router/retriever/answer + <code>llm.tools.openai</code> · <code>docs/assets/langfuse_trace.png</code></sub>
    </td>
    <td width="50%" align="center">
      <img src="https://placehold.co/600x360?text=Architecture" alt="아키텍처" width="100%"><br>
      <sub>아키텍처 다이어그램 · <code>docs/assets/architecture.png</code></sub>
    </td>
  </tr>
</table>

> 📸 위 placeholder는 `docs/assets/`에 실제 이미지를 넣고 각 `<img src>`를 상대경로(예: `docs/assets/chat.png`)로 바꾸면 교체됩니다.

---

## Architecture

```
User → FastAPI → LangGraph Workflow
                      │
                      ├── Router      질문 분류·도메인 라우팅 / 범위 밖 질문 차단
                      ├── Retriever   ① deterministic(기본): Dense(Qdrant) + BM25(OpenSearch) RRF → Reranker → boost
                      │               ② single_agentic(opt-in): LLM tool-calling 에이전트
                      ├── Trust       Evidence Confidence 4축 채점 → 근거 부족 시 재검색(rules-only)
                      └── Answer      Answerability Status 판단 → 5요소/freeform 답변 생성·보류

실행 순서: Router → Retriever → Trust → (근거 부족 시 재검색, max_retries) → Answer
관측: 전 구간 Langfuse trace + 에이전트 운영지표 Prometheus /metrics
```

> **명명 주의 — "노드" vs "에이전트"**: `app/agents/`는 LangGraph 노드 디렉토리명일 뿐, 대부분은 tool을 쓰는 자율 에이전트가 아니다. **tool-calling 에이전트는 single_agentic retriever 하나**다.

| 노드 | LLM | tool | 자율 판단 | 정확한 분류 |
|---|---|---|---|---|
| Router | 1회 | ✗ | 질문 분류/라우팅 | LLM 분류기 |
| Retriever (deterministic, 기본) | ✗ | ✗ | ✗ | 검색 파이프라인(알고리즘) |
| Retriever (single_agentic, opt-in) | 루프 | 3개 | ✅ | **tool-calling 에이전트** |
| Trust | 규칙(재작성 시만 1회) | ✗ | 재검색 사다리(규칙) | 규칙 평가기 |
| Answer | 1회 | ✗ | ✗ | LLM 생성기 |

- **두 가지 검색 전략(retrieval strategy)을 제공** — `deterministic`(기본: 고정 파이프라인, 빠르고 재현 가능)과 `single_agentic`(LLM이 도구를 골라 검색하는 tool-calling 에이전트).
- `RETRIEVER_STRATEGY`(env) 또는 요청 단위로 전환. ([Agentic Retriever](#agentic-retriever) 참고)

## Tech Stack

| 영역 | 기술 |
|---|---|
| Framework | FastAPI, LangGraph |
| LLM | gpt-4o-mini, GPT-4o, Azure OpenAI, self-hosted (Sovereign 선택) |
| Embedding | text-embedding-3-small |
| Reranker | bge-reranker-v2-m3 — 환경별 backend (torch CPU/GPU · ONNX int8 CPU · remote 서비스) |
| Vector DB | Qdrant (dense) |
| Lexical | OpenSearch (BM25, hybrid 융합) |
| DB | PostgreSQL (asyncpg + SQLAlchemy) |
| Cache / Stream | Redis (캐시 + STT 이벤트 stream) |
| Auth | JWT 세션 + Slack OIDC (Sign in with Slack) |
| Observability | Langfuse (trace/score) · Prometheus `/metrics` |
| Infra | EKS (별도 infra 레포: Helm + ArgoCD) |

## Project Structure

```
app/
├── api/            # 엔드포인트 — v1/{chat, asset(s), reports, transcriptions, conversations,
│                   #             monitoring, ingestion, auth, health}, metrics, slack
├── agents/         # LangGraph Agent (router, retriever[deterministic+single_agentic], trust, answer)
├── rag/            # RAG 코어 (embedder, chunker, indexer, hybrid_search, reranker, sources/github)
├── eval/           # 평가 — 골든셋 로더, 지표(agentic_metrics·metrics), RAGAS judge
├── observability/  # Langfuse(langfuse) + 에이전트 운영지표(agent_metrics)
├── auth/           # JWT 세션 · Slack OIDC
├── services/       # 비즈니스 로직 (chat, asset, report, conversation, llm_selector …)
├── db/             # 데이터 접근 (qdrant, opensearch, postgres, redis, confluence)
├── queue/ workers/ # STT 이벤트 outbox/consumer + 보고서 생성 worker
├── storage/        # 오브젝트 스토리지(S3 호환)
├── middleware/     # Request ID, 로깅, 에러 핸들링
└── models/         # Pydantic 스키마 (request, response)
```

## Getting Started

### Prerequisites
- Python 3.11+ (배포 런타임 `python:3.11-slim`)
- [uv](https://docs.astral.sh/uv/) (패키지 매니저)
- Docker & Docker Compose (로컬 인프라)

### Setup

```bash
git clone https://github.com/OnRamp-2026/onramp-api.git
cd onramp-api

pip install uv
uv venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# (선택) 리랭커 — 미설치 시 vector score 순 폴백으로 동작 (아래 '리랭커' 참고)
make install-rerank

cp .env.example .env        # API 키 등 입력
docker compose up -d        # Qdrant, PostgreSQL, Redis, OpenSearch
alembic upgrade head        # DB 마이그레이션
make dev                    # 서버 실행 (http://localhost:8000/docs)
```

### Makefile Commands
```
make dev              개발 서버 (--reload)
make test             전체 테스트
make test-unit        단위 테스트만
make test-cov         커버리지 리포트
make lint             ruff 린트 검사
make format           자동 포맷 + 린트 수정
make typecheck        mypy 타입 체크
make migrate          DB 마이그레이션 적용
make migrate-new      새 마이그레이션 생성
make up / make down   로컬 인프라 (docker compose)
make install          의존성 + pre-commit (1회성)
make install-rerank   리랭커 의존성 (CPU torch + sentence-transformers)
make setup-reranker-onnx  ONNX 리랭커 셋업 (install-onnx + build)
make bench-reranker-onnx  torch vs ONNX(int8) 속도·품질 벤치
make clean            캐시 정리
```

## API Endpoints

> 대화형 문서: 서버 실행 후 `http://localhost:8000/docs` (Swagger UI)
> `/v1/chat` 등 대부분 엔드포인트는 **인증 필수**(#163) — 세션 쿠키 또는 `Authorization: Bearer <token>`.

| 그룹 | 엔드포인트 | 설명 |
|---|---|---|
| Chat | `POST /v1/chat` · `POST /v1/chat/feedback` | 질문→5요소/freeform 답변 · 답변 피드백(👍/👎) score |
| Conversations | `GET /v1/conversations` · `GET /v1/conversations/{id}/messages` · `DELETE …/{id}` | 대화 이력 |
| Asset (자산화) | `POST /v1/asset` · `GET/PATCH /v1/asset/{id}` · `POST /v1/asset/{id}/approve` | 녹취/텍스트→보고서→Confluence 등록 |
| Assets/Reports | `GET /v1/assets` · `DELETE /v1/assets/{id}` · `GET/PATCH /v1/reports/{id}` · `POST /v1/reports/{id}/approve` | 자산 이력·보고서 CRUD/승인 |
| Transcriptions (STT) | `POST /v1/transcriptions` · `GET /v1/transcriptions/{id}` | 업로드/상태 |
| Ingestion | `POST /v1/ingestion/runs` · `GET /v1/ingestion/runs[/current]` | 적재 작업 트리거/조회 |
| Monitoring | `GET /v1/monitoring/overview` · `GET /v1/monitoring/details/{id}` | 운영 관측 집계 |
| Auth | `GET /v1/auth/slack/*` · `POST /v1/auth/dev-token` · (브라우저) `/auth/{login,callback,me,logout}` | Slack OIDC · dev 토큰(게이트) |
| Health / Metrics | `GET /v1/health[/ready]` · `GET /metrics` | 헬스체크 · Prometheus |

### `POST /v1/chat`

Request:
```json
{ "query": "EKS Pod CrashLoopBackOff 어떻게 해결해?", "model": "gpt-4o-mini" }
```

Response (요약):
```json
{
  "answer_format": "structured",
  "answer": { "situation": "...", "cause": "...", "evidence": "...", "solution": "...", "infra_context": "..." },
  "answer_text": "",
  "sources": [ { "title": "...", "url": "...", "score": 0.0 } ],
  "answerability_status": "answerable",
  "answerability_reason": "",
  "domain": "incident",
  "model_used": "gpt-4o-mini",
  "trace_id": "<langfuse-trace-id>",
  "conversation_id": "<id>"
}
```
- `answer_format`: `structured`(5요소, incident) | `freeform`(산문) — 라우터 의도로 분기(#191).
- `answerability_status`: `answerable` / `partially_answerable` / `not_enough_evidence` / `conflicting_evidence` / `outdated_evidence`.
- `trace_id`: Langfuse 활성 시 — `POST /v1/chat/feedback`로 사용자 피드백 score 부착.

## Agentic Retriever

deterministic 검색 전략과 함께, **LLM tool-calling 기반 single_agentic 검색 전략**을 제공한다. LLM이 질의를 보고 도구를 골라 검색·재검색하며, 도구는 서버측 가드로 보호된다.

- 활성: `RETRIEVER_STRATEGY=single_agentic` (env) 또는 요청 state. 기본은 `deterministic` → **끄면 운영 동작 불변**.

**도구 (LLM이 선택)**

| 도구 | 인자 | 용도 | 서버측 가드 |
|---|---|---|---|
| `hybrid_search` | `query` | 전체 출처 Dense+BM25 RRF (기본) | tenant 강제 |
| `hybrid_search_by_source` | `query`, `source`(`github`\|`confluence`) | 출처 제한 검색 | source 화이트리스트 |
| `opensearch_get_document` | `doc_id` | incident 원문 전체 조회 | **incident 도메인 + 앞선 검색의 doc_id만**, 토큰 상한 |

- 정책: **기본 hybrid**, 질의가 콘텐츠 타입을 분명히 드러낼 때만 source 제한(도메인/분포로 추측하지 않음).
- 견고성: tool-call LLM bounded retry → 소진 시 hybrid fallback. tenant/source 서버측 강제.
- 관측: tool 선택·fallback·후보 수를 Langfuse score + Prometheus 카운터로 노출.

설계·측정 근거는 [docs 레포 `Jihong/agentic_*`](https://github.com/OnRamp-2026/docs) 참고.

## Observability

- **Langfuse**: 한 요청 = 한 trace (router/retriever/trust/answer span + LLM generation + token/cost). `LANGFUSE_ENABLED=false`(기본)면 전 구간 no-op(키 없이 기동). Evidence Confidence·gate를 online score로 부착.
- **Prometheus `/metrics`**: worker 큐 지표 + 에이전트 운영지표(`onramp_agent_tool_calls_total{tool=...}` · `_fallbacks_total` · `_retry_steps_total` · `_steps_total`).

## Evaluation

골든셋(`data/eval/`) 기반 평가 하네스로 설계 결정을 데이터로 검증한다.

```bash
# deterministic vs single_agentic A/B (도구선택 정확도·Hit@k·answerability·운영지표)
python scripts/eval_agentic.py --domain incident --repeats 3

# 생성 품질(RAGAS faithfulness/correctness) — [eval] extra 필요
uv pip install -e ".[eval]"
python scripts/eval_generation.py --with-reference
```
- `--repeats N`: LLM 확률성 통제(평균±표준편차).
- ⚠️ `RETRIEVER_STRATEGY` env를 켜고 실행하면 모든 arm이 그 전략으로 오염됨 — 끄고 실행(arm은 요청 state로 제어).

## Environment Variables

```bash
# LLM (Sovereign 선택: model 이름 우선 → LLM_PROVIDER → openai)
OPENAI_API_KEY=sk-...
AZURE_OPENAI_ENDPOINT=...   AZURE_OPENAI_API_KEY=...
DEFAULT_MODEL=gpt-4o-mini

# Retriever (deterministic 기본 / single_agentic opt-in)
RETRIEVER_STRATEGY=deterministic        # | single_agentic

# Vector DB
QDRANT_HOST=localhost   QDRANT_PORT=6333   QDRANT_COLLECTION=onramp

# Hybrid Search — BM25는 '적재 시 색인', HYBRID는 '검색 시 융합'(별개 플래그)
OPENSEARCH_HOST=localhost
BM25_SEARCH_ENABLED=true                 # 적재 시 OpenSearch 청크 BM25 색인
HYBRID_SEARCH_ENABLED=true               # 검색 시 dense+BM25 RRF (off면 dense-only)

# Reranker — 환경별 backend (어느 backend든 실패 시 vector 폴백)
RERANKER_BACKEND=torch                   # torch | onnx | remote
RERANKER_DEVICE=cpu                      # torch 전용: cpu | cuda
# RERANKER_ONNX_DIR=models/bge-reranker-onnx-int8   # onnx
# RERANKER_SERVICE_URL=http://onramp-reranker:8080  # remote

# Database / Cache
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/onramp
REDIS_URL=redis://localhost:6379/0

# Auth (JWT 세션 + Slack OIDC)
AUTH_JWT_SECRET=...
AUTH_ENABLE_SLACK_LOGIN=false
AUTH_DEV_TOKEN_ENABLED=false             # /auth/dev-token 게이트 (운영 false)
SLACK_CLIENT_ID=...   SLACK_CLIENT_SECRET=...

# Observability (Langfuse) — kill-switch, false면 전부 no-op
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=pk-lf-...   LANGFUSE_SECRET_KEY=sk-lf-...   LANGFUSE_HOST=...

# Confluence / GitHub (적재)
CONFLUENCE_BASE_URL=...  CONFLUENCE_API_TOKEN=...  CONFLUENCE_USER_EMAIL=...  CONFLUENCE_SPACE_KEY=OnRamp
GITHUB_TOKEN=...        GITHUB_ORG=OnRamp-2026
```
전체 목록은 [`.env.example`](.env.example) 참고.

## 리랭커 (환경별 backend)

검색 후보를 cross-encoder(`bge-reranker-v2-m3`)로 재정렬한다. **배포 환경에 맞춰 backend를 고른다** — 같은 이미지·코드에서 env(`RERANKER_BACKEND`)로만 전환. **어느 backend든 실패하면 vector score 순으로 폴백**하므로 리랭커가 없거나 죽어도 API는 정상 동작한다.

| 환경 | backend | 설정 |
|---|---|---|
| GPU 노드 (prod) | `torch` (GPU) | `RERANKER_BACKEND=torch` · `RERANKER_DEVICE=cuda` |
| CPU 노드 / 로컬 | `onnx` (int8 경량) | `RERANKER_BACKEND=onnx` · `RERANKER_ONNX_DIR=...` (`make setup-reranker-onnx`) |
| CPU, 의존성 최소 | `torch` (CPU) | `RERANKER_BACKEND=torch` · `RERANKER_DEVICE=cpu` |
| (선택) 분리 운영 | `remote` | 별도 서비스(`onramp-reranker`) — 메모리 격리. `RERANKER_SERVICE_URL=...` |

## 데이터 적재

Confluence·GitHub 원문을 PostgreSQL(원장)·Qdrant(dense)·OpenSearch(BM25/원문)에 적재한다. 같은 스크립트가 로컬·prod에서 동작한다(차이는 env뿐).

```bash
bash scripts/ingest_all_local.sh                                 # 로컬 — 인프라 기동 + 전체 적재
kubectl exec -n onramp deploy/onramp-api -- bash scripts/ingest_all.sh   # prod
python scripts/index_github.py --repos onramp-api gitops ...     # GitHub 단독
python scripts/ingest_status.py                                  # 현황
```
스크립트별 역할·증분/전체·디버깅은 [`docs/ingestion.md`](docs/ingestion.md) 참고.

## STT 자동 보고서 worker

업로드 workflow 이후 STT 이벤트 소비·보고서 생성을 별도 프로세스로 실행한다.
```bash
python -m app.workers.outbox_publisher
python -m app.workers.stt_event_consumer
python -m app.workers.report_generator
```
긴 전사문은 `REPORT_WINDOW_MAX_CHARS`/`REPORT_WINDOW_OVERLAP_CHARS` 기준 구간 추출 후 병합.

## Development

```bash
# 의존성 추가: pyproject.toml 수정 후
uv pip install -e ".[dev]"

# 테스트
pytest                              # 전체
pytest tests/unit/ -v               # 단위
pytest --cov=app --cov-report=html  # 커버리지

# 린트 & 포맷 / 타입
ruff check app/ --fix && ruff format app/
mypy app/

# DB 마이그레이션
alembic revision --autogenerate -m "msg"   # 생성
alembic upgrade head / downgrade -1         # 적용/롤백
```
기여 규칙(브랜치·커밋·PR)은 [`GROUND_RULES.md`](GROUND_RULES.md) 참고.

## Deployment

EKS 배포는 infra 레포에서 Helm + ArgoCD로 관리한다. 이 레포 CI(`Jenkinsfile`)는 빌드→테스트→이미지 푸시까지 담당한다.
```
push → Jenkins → ruff + pytest + mypy → kaniko build → 사내 registry push → GitOps(values) 갱신 → ArgoCD sync
```

## Documentation

| 문서 | 내용 |
|---|---|
| [`GROUND_RULES.md`](GROUND_RULES.md) | 브랜치·커밋·PR 규칙 |
| [`docs/ingestion.md`](docs/ingestion.md) | 데이터 적재 — 저장소·스크립트·로컬/prod·증분 |
| [`docs/local_ingestion.md`](docs/local_ingestion.md) | 로컬 적재 단계별 절차 |
| [`scripts/reranker/README.md`](scripts/reranker/README.md) | (선택) remote 리랭커 운영 |

## Related Repositories

| 레포 | 설명 |
|---|---|
| infra | EKS · Terraform · Helm · 모니터링 · CI/CD |
| confluence-data-crawler | Confluence 일배치 수집 · 청킹 · 임베딩 · Qdrant 적재 |
| onramp-web | Vue 3 챗봇 프론트엔드 |
| docs | 설계·평가 문서 |
