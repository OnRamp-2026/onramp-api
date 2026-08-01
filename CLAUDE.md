# CLAUDE.md — onramp-api 개발 운영 규칙

> Claude Code가 자동으로 읽는다. 이 레포에서 AI는 **통제 가능한 개발 프로세스**로 동작한다.
> 이 파일은 150~200줄 이내 유지. 상세는 `docs/dev-workflow/`, 팀 규칙은 `GROUND_RULES.md`.

## 프로젝트 개요
- **무엇**: 자연어 질문에 구조화된 답변을 생성하는 **RAG 백엔드 서버** (OnRamp, Cortex 팀 · track C·D)
- **기술 스택**: Python 3.11 · FastAPI · LangGraph/LangChain · Qdrant(벡터) · PostgreSQL(asyncpg/SQLAlchemy 2.0 async) · Redis · Alembic · Langfuse(LLMOps)
- **아키텍처**: 계층형 (`app/` = API 라우터 → 서비스 → 리트리버/그래프 → 저장소). 배포는 Docker + Jenkins.
- **주요 명령어**
  | 목적 | 명령 |
  |---|---|
  | 로컬 실행 | `make dev` (uvicorn :8000) |
  | 테스트 | `make test` · `make test-unit` · `make test-integration` · `make test-cov` |
  | 타입체크 | `make typecheck` (mypy `app/`) |
  | lint | `make lint` (ruff check + format --check) · 수정은 `make format` |
  | 마이그레이션 | `make migrate` · `make migrate-new` · `make migrate-down` |
  | 검색 평가 | `make eval` · **`make eval-gate`** (골든셋 게이트) |
  | 생성 평가 | `make eval-gen` (RAGAS LLM-judge, 비결정 → 비차단) |
  | 로컬 인프라 | `make up` / `make down` / `make logs` |

## 프로젝트 이해 방식
- 낯선 코드는 **먼저 리서치**한다(구조·데이터 흐름·기존 패턴). 추측 금지 — 근거를 코드에서 확인.
- 변경 전 유사 구현(참조)을 찾아 그 패턴을 따른다. **새 라이브러리는 함부로 추가하지 않는다** —
  `pyproject.toml`에 버전 고정 사유가 주석으로 달린 의존성이 여럿 있다(ragas·transformers·langchain-community). 건드리기 전에 주석을 읽는다.
- RAG 관련 변경은 코드만으로 판단하지 않는다 — **지표로 판단한다**(아래 검증).

## 구현 전 계획 승인 (핵심)
- **바로 구현하지 않는다.** `Research → Plan → Approve → Implement` 순서를 따른다(→ `docs/dev-workflow/workflow.md`).
- `/research-plan`으로 리서치 + 계획서(`docs/plans/<YYYY-MM-DD>-<기능>.md`)를 만든다.
- **사람이 승인한 계획만** `/implement-from-plan`으로 구현한다. 계획에 없는 변경은 하지 않는다.
- 예외: 오탈자·1줄 버그픽스 등 자명한 변경은 계획 생략 가능. 단 검증은 생략하지 않는다.

## 디버깅 — 추측 금지
- 버그·테스트 실패·예상 못한 동작을 만나면 **수정안을 내기 전에** `/systematic-debugging`.
- **철칙: 근본 원인 조사 없이 수정 없음.** 증상만 고치는 건 실패다.
- 3번 고쳐도 안 되면 → 4번째 수정을 시도하지 말고 **아키텍처를 의심하고 사람에게 보고**한다.
- RAG 품질 저하는 코드부터 보지 않는다 — `make eval-gate`로 **어느 지표가** 떨어졌는지부터.

## 검증 필수
커밋 전 반드시 통과 (→ `docs/dev-workflow/verification-loop.md`):
1. `make test` — 관련 테스트 통과 (pytest, `asyncio_mode=auto`, 커버리지 term-missing 자동)
2. `make typecheck` — mypy 오류 0
3. `make lint` — ruff check + format 통과
4. **RAG 검색 로직(리트리버·청킹·리랭커·프롬프트)을 건드렸으면 `make eval-gate`** — 골든셋 지표 하락 여부 확인. 지표가 안 나오면 그 변경은 폐기 후보다.
5. DB 스키마 변경이면 `make migrate` / `make migrate-down` 왕복 확인.
- 검증할 수 없는 것은 커밋하지 않는다.

## 팀 Ground Rule 준수 (`GROUND_RULES.md` — 위반 시 PR 반려)
- 브랜치: `feat/#N` · `fix/#N` · `chore/#N` (main 직접 push 금지, PR 필수, Squash and Merge)
- 커밋 메시지: `<타입>: <설명> (#이슈번호)` — 타입 `feat`/`fix`/`docs`/`chore`
- **`Co-Authored-By` 트레일러 절대 금지** — Claude가 기본으로 붙이려 하므로 매 커밋 확인할 것.
- `--no-verify` · (push된 커밋의) `--amend` · main 대상 `git push --force` 금지
- PR 본문에 `Close #N` 포함, 확인 방법 명시

## 금지 행동
- 자동 `git push`, 자동 PR 생성, 자동 대량 파일 수정, 과한 알림 hook 금지. **push/PR은 사람 승인 후.**
- `--dangerously-skip-permissions` 사용 금지(격리 환경 제외).
- **`.env` 읽기/수정/노출 금지** (레포 루트에 실키가 있는 실파일이다). 설정 확인은 `.env.example`로.
- 근거 없는 추측·과장 금지. 테스트가 실패하면 실패했다고 말한다.

## 커밋 / PR 전 체크리스트
- [ ] `make test` · `make typecheck` · `make lint` 통과
- [ ] (RAG 변경 시) `make eval-gate` 지표 확인
- [ ] 변경 요약(무엇을·왜)
- [ ] `.env`·시크릿 미접근 / 미커밋
- [ ] 계획서 대비 범위 이탈 없음
- [ ] 커밋 메시지에 `(#N)` 있고 `Co-Authored-By` 없음
- [ ] **push/PR은 사람 승인 후**

## 워크플로우 명령 (Claude Code 단독 운영 — Codex 사용 안 함)
- `/research-plan <작업>` → 리서치 + 계획서 (구현 X)
- 구현 — 규모로 갈린다:
  - 태스크 3개 이하 / 서로 얽힘 → `/implement-from-plan <계획서 경로>` (메인 세션이 직접)
  - 태스크 4개 이상 / 대체로 독립 → **`/subagent-driven-development`** (태스크별 신규 서브에이전트 + 태스크별 리뷰어)
- **`/systematic-debugging`** → 버그·테스트 실패·예상 못한 동작. **근본 원인 찾기 전 수정 금지.**
- `/test-plan` · `/tdd` → 테스트 · `/refactor-clean` → 정리
- `/verify` → 테스트·타입체크·lint 실행 · `/commit` → 커밋(push X) · `/pr` → PR 초안 · `/context-check` → 상태 점검
- `/handoff` → 인수인계 문서(`docs/handoff/<날짜>.md`)

### 리뷰는 2단
1. `/code-review` (빌트인) — 일반 리뷰. 필요하면 `/security-review`.
2. `/review-checklist` — 이 레포 고유 항목(RAG 지표·의존성 고정·계획 이탈).
- 더 독립적인 판단이 필요하면 **`code-reviewer` 서브에이전트**에 위임(격리 컨텍스트라 구현 대화의 편향이 없음). 설계 대안은 `planner`, 테스트 계약은 `tdd-guide`. → `docs/dev-workflow/roles.md`

## 알려진 함정 (누적 — AI가 반복 실수하면 여기에 추가)
- **커밋에 `Co-Authored-By`가 자동으로 붙는다** → 팀 규칙 위반. 커밋 전 메시지 확인.
- `pyproject.toml`의 버전 상한(`ragas<0.3`, `transformers<5`, `langchain-community<0.4`)은 의도된 고정이다. 임의 상향 금지 — 주석의 사유 확인.
- mypy `python_version=3.11` + numpy 스텁 `follow_imports=skip` 설정은 CI(3.12)와의 충돌 회피용. 임의 변경 금지.
- 생성 평가(`make eval-gen`)는 LLM-judge라 비결정적 → **CI 게이트로 쓰지 않는다**(nightly·수동).
- (여기에 반복 실수를 계속 추가)

---
- 팀 규칙: `GROUND_RULES.md` · 전체 흐름: `docs/dev-workflow/workflow.md` · 검증: `docs/dev-workflow/verification-loop.md` · 역할: `docs/dev-workflow/roles.md`
