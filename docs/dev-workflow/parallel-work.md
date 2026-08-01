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
| `.env` | **안 따라옴** | gitignore. 심링크 안 하면 앱이 아예 안 뜬다 |
| `.venv/` | **안 따라옴** | gitignore. worktree마다 `make install` 필요 |

## 병렬 안전 등급

| 명령 | 병렬 | 이유 |
|---|---|---|
| `make test-unit` | **안전** (단 `.env` 주의) | in-memory sqlite · `FakeRedis` · `httpx.MockTransport`라 서로 밟을 공유 상태가 없다. **다만 결과가 `.env`에 좌우된다** — 아래 참조 |
| `make lint` · `make typecheck` | **안전** | 로컬 파일만 |
| `make eval` · `make eval-gate` | **안전** | 공유 Qdrant/OpenSearch를 **읽기만** 한다. 쓰기는 `--write-baseline` 있을 때 로컬 `data/eval/baseline.json`뿐이고 두 타깃 다 안 넘긴다 (`Makefile`). 단 **API 비용은 배로** |
| `make test` · `make test-integration` | **직렬** | Qdrant 컬렉션을 서로 지운다 — 아래 |
| `make migrate` | **직렬** | 공유 Postgres 하나. 두 브랜치가 각자 마이그레이션을 만들면 head가 갈린다 |
| `make dev` | **직렬** | `:8000` 하드코딩 |
| `make eval-dataset-push` | **직렬** | 고정 이름 Langfuse 데이터셋을 덮어쓴다 (`scripts/eval_push_dataset.py`) |
| `make seed-monitoring-local` | **직렬** | 실 Postgres 행을 tenant 기준으로 지운다 (`scripts/seed_monitoring_local.py`) |

## 왜 통합 테스트가 충돌하나

컬렉션명이 **고정**이고, setup과 teardown **양쪽에서** 삭제한다:

```python
# tests/integration/test_qdrant_index.py:10,21-24
COLLECTION = "onramp_itest"                                    # 실행별 구분 없음
if COLLECTION in {...}: client.delete_collection(COLLECTION)   # setup에서 삭제
yield client
client.delete_collection(COLLECTION)                           # teardown에서 또 삭제
```

worktree A가 검색 중인 컬렉션을 worktree B의 setup이 지운다. 반대로 B가 심은 데이터를 A의 teardown이 날린다. 같은 패턴이 3곳:

| 파일 | 컬렉션명 |
|---|---|
| `tests/integration/test_qdrant_index.py:10` | `onramp_itest` |
| `tests/integration/test_retrieval.py:16` | `onramp_c2_itest` |
| `tests/integration/test_eval_retrieval.py:18` | `onramp_eval_itest` |

증상이 헷갈린다 — 실패 메시지는 "컬렉션이 없다"라서 **자기 코드 버그로 보인다.** 통합 테스트가 이유 없이 깨지면 다른 worktree가 도는지부터 확인할 것.

> **참고:** Postgres·Redis·OpenSearch는 테스트가 건드리지 않는다. DB는 전부 `sqlite+aiosqlite:///:memory:`, Redis는 인라인 `FakeRedis`, OpenSearch는 `httpx.MockTransport`다. `tests/conftest.py`의 `client` 픽스처는 `ASGITransport`라 lifespan조차 안 돈다. **충돌은 Qdrant 하나뿐이다.**

## `.env`가 유닛 테스트 결과를 바꾼다 (함정)

설정은 `pydantic-settings`가 **현재 디렉터리의 `.env`를 읽어** 만든다. 유닛 테스트도 이 설정을 쓰므로, **`.env`가 있느냐 없느냐로 결과가 달라진다.**

실측(2026-08-01, `chore/#299` 기준):

| 위치 | `test_observability_langfuse.py` + `test_retriever_node.py` |
|---|---|
| 원본 디렉터리 (`.env` 있음) | **7 실패** — `test_langfuse_disabled_by_default`가 `assert True is False` 등 |
| 새 worktree (`.env` 없음) | **34개 전부 통과** |

CI는 `.env` 없이 돌아서 통과한다(`.github/workflows/ci.yaml`). 즉 **"내 로컬에서만 깨지는" 유닛 테스트는 대개 내 `.env` 탓**이지 코드 탓이 아니다. 반대로 `.env`를 심링크한 worktree는 원본과 똑같이 7개가 깨진다.

→ 유닛 테스트가 로컬에서만 깨지면 **먼저 `.env` 유무를 의심할 것.** 근본 해결(테스트가 설정을 override)은 아래 "알려진 한계".

## 새 worktree 체크리스트

```bash
# 1) 생성 (브랜치명은 GROUND_RULES §1-2)
git worktree add ../onramp-301 -b "feat/#301"
cd ../onramp-301

# 2) 의존성
make install

# 3) 유닛 테스트·lint·타입체크는 .env 없이 (CI와 동일 조건 = 깨끗한 기준선)
make test-unit && make lint && make typecheck

# 4) .env 연결 — 앱 실행·통합 테스트·평가에 필요할 때만
#    (연결하면 위 유닛 테스트 7개가 원본과 똑같이 깨진다)
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
