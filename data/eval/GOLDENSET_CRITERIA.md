# 골든셋 생성 기준 (재설계, 2026-06-17)

현재 로컬 적재 코퍼스를 대상으로 골든셋을 **재구성**하기 위한 설계 기준.
실제 생성은 이 기준 확정 후 별도 단계에서 수행한다(본 문서는 "무엇을·왜·얼마나"를 고정).
운영 포맷·필드 정의는 [`README.md`](README.md)를 따르며, 본 문서는 **샘플링·구성 기준**만 정의한다.

관련성 단위는 **chunk_id**(`{page_id}_{idx:03d}`, Qdrant payload `chunk_id`).

---

## 1. 코퍼스 스냅샷 (재현 기준)

측정 당시 적재 상태. baseline 재고정 시 `baseline.json`의 `config.corpus`와 일치해야 한다.

| 저장소 | 카운트 |
|---|---|
| Qdrant `onramp` (dense 청크) | **7,799 points** |
| Postgres `chunk_registry` | 7,799 rows |
| Postgres `source_document` | 1,257 docs (confluence 909 + github 350) |
| OpenSearch `onramp-chunks` | 7,803 docs |

> README의 #75 기록(864p / 5,731청크)은 **stale**. GitHub 소스 추가로 코퍼스가 다시 커졌다.

### 1-1. 도메인 × 소스 청크 분포 (재설계의 핵심 입력)

| 도메인 | github(PR/이슈) | confluence(운영문서) | **합계** | 코퍼스 비중 |
|---|---:|---:|---:|---:|
| manual | 200 | 5,279 | **5,479** | 70.3% |
| planning | 1,120 | 144 | **1,264** | 16.2% |
| api_reference | 170 | 445 | **615** | 7.9% |
| meeting_note | 0 | 364 | **364** | 4.7% |
| incident | 75 | 2 | **77** | 1.0% |
| **합계** | 1,565 | 6,234 | **7,799** | 100% |

**관찰 (생성 기준에 직접 반영):**
- 코퍼스가 **manual에 70% 쏠림**, incident는 1%(77청크)뿐 — 순수 균등/순수 비례 둘 다 부적합.
- `incident` 77청크 중 **75개가 github PR/이슈**(confluence 운영 incident는 2개). `planning`도 1,264 중
  **1,120(89%)이 github PR**. → 이 두 도메인의 골든셋은 source를 구분하지 않으면 사실상 "PR 검색"을 측정한다.
- `meeting_note`는 100% confluence. `manual`은 사실상 confluence(96%).

---

## 2. 도메인 샘플링 기준 — 층화(floor) + 코퍼스 비례

분포 앵커 = **코퍼스 비중**(질의 로그 미사용으로 결정). 단 순수 비례는 소수 도메인 측정 불가이므로,
**도메인별 최소 보장(floor)** 위에 잔여 예산을 코퍼스 비중에 비례 배분한다.

```
domain_quota(d) = FLOOR + round( (TOTAL_ANSWERABLE - FLOOR*5) * corpus_share(d) )
```

- `FLOOR` = 도메인당 최소 문항. **확정: 8.** 소수 도메인(incident)도 per-domain 측정이 가능한 최소선.
- 잔여(`TOTAL_ANSWERABLE - FLOOR*5`)는 코퍼스 비중대로 → manual에 가산, aggregate가 실분포 근사.

### 배분 (single-hop answerable = **50 확정**, FLOOR = **8 확정**)

| 도메인 | floor | +비례(10×share) | **할당** |
|---|---:|---:|---:|
| manual | 8 | 7 | **15** |
| planning | 8 | 2 | **10** |
| api_reference | 8 | 1 | **9** |
| meeting_note | 8 | 0 | **8** |
| incident | 8 | 0 | **8** |
| **합계** | 40 | 10 | **50** |

### 리포트 규칙 (필수)
검색 지표는 **macro-average**(도메인 동일가중 — 약자 도메인 가시화)와
**micro-average**(문항가중 = 실분포 근사)를 **둘 다** 출력한다. 단일 aggregate만 보면 manual 편향에 가려진다.

---

## 3. 티어 구성 기준

| 티어 | qid | 정의 | answerable | 측정 대상 |
|---|---|---|---|---|
| single-hop | `d0xx` | 청크 1개가 정답 근거 | ✅ | 기본 Hit/Recall |
| multi-domain | `m0xx` | 정답이 2~3 도메인 다른 페이지에 걸침 | ✅ | 도메인 교차 recall (#65) |
| multi-hop | `h0xx` | 같은 페이지 인접 청크 2~3개 종합 | ✅ | Recall과 Hit 분리 |
| confusable | `c0xx` | 벡터 이웃 많은 타깃 청크 저격 | ✅ | 리랭커 변별력 |
| near-miss | `n0xx` | 도메인 내 주제지만 코퍼스에 답 없음 | ❌ | τ answerability 변별 |
| scope-out | `d0xx` | "점심 메뉴"류 범위 밖 | ❌ | Router 차단 |

**구성 원칙:**
- single-hop의 도메인 배분은 §2 공식을 따른다.
- answerable : unanswerable 비율은 진단 목적상 **약 70:30** 유지(near-miss가 변별력 핵심).
- 각 answerable 문항은 §4 pooling 검수로 라벨 누락(중복 문서로 인한 과소평가)을 보완한다.

### 확정 목표 구성 (전면 재생성)

| 티어 | qid | 문항 | answerable |
|---|---|---:|:---:|
| single-hop (§2 배분) | `d0xx` | **50** | ✅ |
| multi-domain | `m0xx` | **6** | ✅ |
| multi-hop | `h0xx` | **8** | ✅ |
| confusable | `c0xx` | **10** | ✅ |
| near-miss | `n0xx` | **18** | ❌ |
| scope-out | `d0xx` | **10** | ❌ |
| **합계** | | **102** | answerable 74 / unanswerable 28 (73:27) |

---

## 4. GitHub 소스 처리 — **확정: 분리하지 않음**

github(PR/이슈)와 confluence(운영문서)를 **구분하지 않고 코퍼스 그대로** 샘플링한다.
source 필터·전용 티어 없음 → 생성 스크립트에 source 조인 불필요(payload에 source 없어도 무방).

- incident/planning이 github PR 위주라는 점은 **인지된 특성으로 수용**(별도 보정 안 함).
- github 청크의 도메인 분류(`fix:`→incident, `feat:`→planning)는 **적절한 것으로 판단** — 재분류·신뢰도 점검 생략.
- §2 도메인 quota는 source 무관하게 도메인 청크 풀 전체에서 뽑는다.

---

## 5. 생성 파이프라인 매핑

| 단계 | 스크립트 | 본 기준에서 필요한 변경 |
|---|---|---|
| 초안 생성 | [`scripts/bootstrap_golden.py`](../../scripts/bootstrap_golden.py) | `sample_per_domain`을 §2 floor+비례 quota로 교체 (source 필터 불필요, §4). **생성 프롬프트 §5-A 개정 반영** + multi-domain 모드 신설(§5-A.5) |
| pooling 검수 | [`scripts/pool_candidates.py`](../../scripts/pool_candidates.py) | 변경 없음 — top-10 후보로 라벨 누락 보완 |
| qrels 검증 | [`scripts/validate_qrels.py`](../../scripts/validate_qrels.py) | chunk_id 현존성 검증(현재 기존 99개 전부 유효) |
| τ 재보정 | [`scripts/calibrate_answerability.py`](../../scripts/calibrate_answerability.py) | 확정 후 재실행 |

> 현재 `bootstrap_golden.py`의 `sample_per_domain`은 **도메인 균등**(`per_domain` 동일)이라 §2와 불일치 —
> 재설계의 주 변경점.

---

## 5-A. 프롬프트 개정 사항 (생성 전 적용)

현재 생성 프롬프트([`bootstrap_golden.py`](../../scripts/bootstrap_golden.py) 4종 +
[`bootstrap_gt_answers.py`](../../scripts/bootstrap_gt_answers.py))에 대한 평가셋 품질 피드백.
ROI 순서: **공통 1·2·4 → 구조 5 → 모드별**.

### 공통 (전 질문 생성 프롬프트)

1. **지시대명사 금지 (라벨 무효화 차단)** — "이 문서/위 조각/해당 설정" 류 표현 시
   원문 없이 검색 불가한 질문이 됨. → *"질문은 원문 조각 없이 단독 성립해야 하며 지시 표현 금지"* 명시.
2. **식별자 보존 vs paraphrase 균형** — 무조건 "문구 베끼지 말 것"이 `SymLinksIfOwnerMatch`·
   `DD_TRACE_PROPAGATION_STYLE` 같은 앵커 토큰까지 제거해 부자연·검색불가를 유발.
   → *"고유명사·명령어·설정 키·에러코드는 원문 그대로, 주변 자연어만 바꿔라"*.
3. **저품질 청크 가드** — 목차/순수 코드/마스킹(`[MASKED_...]`)/헤더뿐인 청크도 무조건 질문 생성.
   → *"답할 실질 내용 없으면 빈 query 반환"*.
4. **자가 검증 신호** — 현재 `query`만 반환해 청크 내 답 존재를 자동 확인 불가.
   → `{"query":…, "answer_span":"근거 문장", "answerable_from_chunk":true}`로 받아 pooling 전 싸게 필터.

### 모드별

- **near-miss (최고 위험)** — LLM은 청크 1개만 보고 7,799청크 전체의 답 부재를 알 수 없어
  **false-unanswerable**이 구조적. (a) 코퍼스가 구조적으로 안 담는 정보 유형(가격·SLA 수치·담당자
  실명·내부 일정)으로 유도, (b) **생성 후 `pool_candidates.py` 검색 강제** — top-k에 진짜 정답
  청크 없을 때만 채택(워크플로에 박을 것, 프롬프트만으론 불가).
- **multi-hop** — "모두 종합" 강제력 약함. → *"조각1·조각2의 사실을 각각 써야만 답 완성,
  한 조각만으로 답되면 실패. 단 복합질문 아닌 자연스러운 한 문장"*. span=2 인접 청크가 한 문장의
  연속이면 종합이 인위적 — 주의.
- **confusable (설계 양호)** — *"타깃 문서에 실제 적힌 내용만 근거로, 유사 문서엔 없는 식별 가능한
  디테일을 물어라"* 한 줄로 라벨 유효성 보강.

### 구조적 공백

5. **multi-domain(`m0xx`) 생성 모드 부재** — `bootstrap_golden.py`엔 single/multi-hop/near-miss/
   confusable 4모드뿐. §3의 `m0xx` 6문항을 만들 모드가 없음 → **multi-domain 모드 신설**(서로 다른
   도메인 청크 2~3개를 정답으로 묶음) **또는 수기 작성으로 명시**.
6. **unanswerable 시드 하드코딩 3개** ([`bootstrap_golden.py`](../../scripts/bootstrap_golden.py)
   `_UNANSWERABLE_SEEDS`) — scope-out 10문항인데 3개뿐이라 반복. → 시드 풀 확장 + **그럴듯한
   도메인 밖**(예: Apache/Datadog 코퍼스에 "AWS Lambda 요금제")을 섞어 라우터 차단 난이도 상향.

### GT 답변 ([`bootstrap_gt_answers.py`](../../scripts/bootstrap_gt_answers.py), 대체로 양호)

- context `[:1200]` 절단 — 정답 문장이 뒤에 있으면 잘림(상한 상향 검토).
- 근거가 질문에 답 불충분한 경우(질문 드리프트) 대비책 없어 지어냄. → *"불충분하면
  `{"ground_truth_answer":""}` 반환"*으로 불량 문항 표면화.

---

## 6. 확정 절차

**재구성 범위 확정: 전면 재생성** — 기존 114문항(qrels chunk_id 99개 전부 현존, stale 0)을 폐기하고
§3 목표 구성(102문항)을 처음부터 생성한다. 기존 `queries.jsonl`/`qrels.jsonl`은 백업 후 교체.

1. `bootstrap_golden.py`의 `sample_per_domain`을 §2 floor+비례 quota로 교체 + **생성 프롬프트 §5-A 개정**.
2. 모드별 초안 생성(single 50 + multi-hop 8 + confusable 10 + near-miss 18), scope-out 10·multi-domain 6 구성
   (multi-domain은 §5-A.5에 따라 모드 신설 또는 수기).
3. pooling → 팀 검수(질문 자연스러움·라벨 정확성·near-miss는 §5-A대로 `pool_candidates.py` 답 부재 강제) → `_draft` 제거.
4. baseline 재고정(`--write-baseline`) + τ 재보정. `config.corpus`로 §1 스냅샷과 일치 확인.

---

## 확정 결정 요약

- ✅ `TOTAL_ANSWERABLE` = **50** (single-hop), `FLOOR` = **8** (§2)
- ✅ GitHub: **분리 안 함, 코퍼스 그대로 생성** (§4)
- ✅ 재구성 범위: **전면 재생성** (§6)
- ✅ github 도메인 분류: **적절 판단 — 점검 생략** (§4)
- 🎯 목표: **102문항** (answerable 74 / unanswerable 28)
