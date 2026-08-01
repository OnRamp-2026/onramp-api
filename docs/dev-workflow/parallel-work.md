# 병렬 작업 (Git Worktree)

> 이슈 여러 개를 동시에 진행할 때. **이슈 1개 = `feat/#N` 브랜치 1개 = worktree 1개** (`GROUND_RULES.md` §1-2).
> worktree는 파일과 git 상태만 격리한다. **인프라는 공유된다** — 아래 "격리되는 것 / 안 되는 것"을 먼저 읽을 것.

## 기본 사용법
```bash
# 이슈별 독립 워크트리 생성
git worktree add ../onramp-301 -b "feat/#301"
git worktree add ../onramp-302 -b "feat/#302"

# 각 워크트리에서 별도 세션 실행 (동시)
cd ../onramp-301 && claude
cd ../onramp-302 && claude
```

clone과 달리 `.git` 오브젝트 저장소를 **공유**한다 → 디스크를 다시 안 먹고, `git fetch` 한 번이면 전부 최신이고, 한쪽에서 만든 브랜치가 다른 쪽에서 바로 보인다.

Orca ADE 같은 도구는 이 과정(worktree 생성 · 터미널/에디터 연결 · 에이전트 기동 · 정리)을 자동화한다. **자동화되는 건 잡일이지 아래의 공유 자원 문제가 아니다.**

## 격리되는 것 / 안 되는 것

| 항목 | 격리? | 비고 |
|---|---|---|
| 소스 파일 · 브랜치 · 커밋 | **O** | worktree의 본래 목적 |
| `.sdd/` (SDD 작업 공간) | **O** | `git rev-parse --show-toplevel` 기준이라 worktree마다 자동 분리 |
| **Qdrant `localhost:6333`** | **X** | **통합 테스트 충돌의 원인** — 아래 참조 |
| OpenSearch `:9200` · Postgres `:5432` · Redis `:6379` | **X** | `make up`으로 띄운 컨테이너는 머신 전체에 하나 (`docker-compose.yml`) |
| `make dev`의 `:8000` | **X** | `Makefile`에 하드코딩. 둘째 worktree는 `uvicorn app.main:app --port 8001`로 직접 실행 |
| OpenAI API 비용 | **X** | worktree 3개 = 임베딩/LLM 호출 3배 |
| `.env` | **안 따라옴** | gitignore. 앱 실행·통합 테스트에 필요하면 심링크 |
| `.venv/` | **안 따라옴** | gitignore. worktree마다 `make install` 필요 |

## 병렬 안전 등급

| 명령 | 병렬 | 이유 |
|---|---|---|
| `make test-unit` | **안전** | in-memory sqlite · `FakeRedis` · `httpx.MockTransport`라 공유 상태가 없다. `.env`도 차단된다(`tests/unit/conftest.py`) |
| `make lint` · `make typecheck` | **안전** | 로컬 파일만 |
| `make test` · `make test-integration` | **안전** | Qdrant 테스트 컬렉션이 실행별로 네임스페이스된다 — 아래 |
| `make eval` · `make eval-gate` | **안전** | 공유 Qdrant/OpenSearch를 **읽기만** 한다. 쓰기는 `--write-baseline` 있을 때 로컬 `data/eval/baseline.json`뿐이고 두 타깃 다 안 넘긴다 (`Makefile`). 단 **API 비용은 배로** |
| `make migrate` | **직렬** | 공유 Postgres 하나. 두 브랜치가 각자 마이그레이션을 만들면 head가 갈린다 |
| `make dev` | **직렬** | `:8000` 하드코딩. 둘째는 `uvicorn app.main:app --port 8001`로 |
| `make eval-dataset-push` | **직렬** | 고정 이름 Langfuse 데이터셋을 덮어쓴다 (`scripts/eval_push_dataset.py`) |
| `make seed-monitoring-local` | **직렬** | 실 Postgres 행을 tenant 기준으로 지운다 (`scripts/seed_monitoring_local.py`) |

## 통합 테스트 격리 — 실행별 컬렉션 네임스페이스

Qdrant는 머신에 하나뿐이라 worktree를 나눠도 공유된다. 테스트가 setup·teardown **양쪽에서** `delete_collection` 하므로, 컬렉션명이 고정이면 동시 실행 시 서로를 지운다. 그래서 실행별 접미사를 붙인다:

```python
# tests/integration/test_qdrant_index.py
_NS = os.getenv("ONRAMP_TEST_NS") or str(os.getpid())
COLLECTION = f"onramp_itest_{_NS}"
```

같은 방식이 3곳: `test_qdrant_index.py`(`onramp_itest`) · `test_retrieval.py`(`onramp_c2_itest`) · `test_eval_retrieval.py`(`onramp_eval_itest`).

기본값은 **PID**라 아무 설정 없이도 병렬 실행이 안전하다. CI 잡처럼 이름을 고정하고 싶으면 `ONRAMP_TEST_NS`로 덮어쓴다:

```bash
ONRAMP_TEST_NS=ci-$GITHUB_RUN_ID make test-integration
```

**실측 검증** (Qdrant 기동 상태에서 두 세션 동시 실행):

| | 결과 |
|---|---|
| 네임스페이스 이전 (고정명) | 두 세션 **모두 7 실패** |
| 네임스페이스 이후 | 두 세션 **모두 7 통과**, 잔류 컬렉션 없음 |

> **참고:** Postgres·Redis·OpenSearch는 테스트가 건드리지 않는다. DB는 전부 `sqlite+aiosqlite:///:memory:`, Redis는 인라인 `FakeRedis`, OpenSearch는 `httpx.MockTransport`다. `tests/conftest.py`의 `client` 픽스처는 `ASGITransport`라 lifespan조차 안 돈다. **테스트가 쓰는 실 서비스는 Qdrant 하나뿐이다.**

## 유닛 테스트는 `.env`를 읽지 않는다

`Settings`는 pydantic-settings라 `Settings()`를 인자 없이 부르면 cwd의 `.env`를 읽어 **코드 기본값 위에 덮어쓴다.** 유닛 테스트는 "코드에 적힌 기본값"을 검증하므로 이게 섞이면 `.env`를 채워둔 로컬에서만 테스트가 깨진다(CI는 `.env`가 없어 통과).

`tests/unit/conftest.py`의 autouse 픽스처가 유닛 테스트 동안 `env_file`을 `None`으로 막는다. 덕분에 **`.env` 심링크를 걸든 말든 유닛 테스트 결과가 CI와 같다.**

환경변수(`os.environ`)는 차단하지 않는다 — CI가 의도적으로 주입하는 경로이고 `client` 픽스처도 `DEBUG`를 환경변수로 세팅한다. 통합 테스트에는 적용하지 않는다(실 자격증명·호스트가 필요).

→ 유닛 테스트가 로컬에서만 깨지면 이제 `.env`가 아니라 **셸에 export된 환경변수**를 의심할 것.

## 새 worktree 체크리스트

```bash
# 1) 생성 (브랜치명은 GROUND_RULES §1-2)
git worktree add ../onramp-301 -b "feat/#301"
cd ../onramp-301

# 2) 의존성
make install

# 3) 확인
make test-unit && make lint && make typecheck

# 4) .env 연결 — 앱 실행·통합 테스트·평가에 필요할 때만
ln -s ~/onramp-api/.env .env
```

**요약:** `.env`는 **앱을 띄우거나 통합/평가를 돌릴 때만** 연결한다. 코드만 짜고 유닛 테스트로 확인하는 worktree라면 없는 편이 낫다.

## 정리

```bash
git worktree remove ../onramp-301        # 변경 없을 때. 있으면 --force
git worktree prune                       # 폴더를 손으로 지웠을 때 메타데이터 정리
git worktree list                        # 현황
git branch -d "feat/#301"                # 머지됐으면 브랜치도 (GROUND_RULES §2 [7])
```

## 주의 (통제 우선)
- 병렬은 **독립적인 작업**에만. 서로 얽힌 변경은 순차로.
- 각 세션도 계획 승인·검증 루프를 똑같이 지킨다(병렬이라고 검증 건너뛰지 않기).
- push는 각 브랜치별로 사람이 승인.
- 통합 테스트·마이그레이션은 **한 번에 하나의 worktree에서만.**

## 서브에이전트 병렬
- 대규모 분석/수정은 기능별 서브에이전트(`.claude/agents/`)에 나눠 병렬 실행.
- 계획서의 태스크를 병렬로 돌리려면 `/subagent-driven-development` — 단 이 스킬은 **구현 서브에이전트를 병렬로 띄우지 않는다**(충돌 방지). 병렬성은 worktree 단위로 얻는다.

## 알려진 한계
- **통합 테스트 병렬화가 안 된다.** 컬렉션명을 실행별로 네임스페이싱(PID/UUID 접미사)하면 해결되지만 현재 그런 코드가 없다. 필요해지면 별도 이슈로.
- **유닛 테스트가 `.env`에 오염된다.** 테스트가 설정을 명시적으로 override하지 않아 로컬 `.env` 값이 새어 든다(위 "함정" 참조). 테스트 픽스처에서 `Settings`를 주입하거나 `pytest` 실행 시 `_env_file=None`을 강제하면 해결된다. 별도 이슈 후보 — 지금은 "로컬에서만 깨지면 `.env` 의심"으로 대응.
- worktree마다 `.venv/`를 따로 만들면 디스크를 먹는다. 공유하려면 `VIRTUAL_ENV`를 원본으로 지정하되, 브랜치별 의존성이 다르면 깨진다.
