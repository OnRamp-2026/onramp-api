# 검색 평가 골든셋 (`data/eval/`)

우리 코퍼스(Qdrant 색인분) 위에서 검색 품질을 **결정론적으로** 측정하기 위한 정답지.
관련성 단위는 **chunk_id** (Qdrant payload의 `chunk_id`, 형식 `{page_id}_{idx:03d}`).

## 파일

| 파일 | 내용 |
|---|---|
| `queries.jsonl` | 질문 + 메타 (1줄=1질문) |
| `qrels.jsonl` | 질문별 정답 chunk_id 라벨 (1줄=1질문) |
| `baseline.json` | 검색 베이스라인 수치 (회귀 기준, CLI가 생성) |
| `baseline.single_label.json` | **멀티라벨 재색인 전 single-label 스냅샷**(A/B 비교 고정본, 재현 메타 포함). `--gate`는 `baseline.json`을 쓰며 이 파일은 수동 비교용 (#49 / #86) |
| `gen_report.json` | 생성 평가(RAGAS) 최근 리포트 — **gitignore**(LLM-judge 비결정 → 로컬/CI 아티팩트로만) |

### `queries.jsonl`
```json
{"qid":"q001","query":"...","domain":"incident","gold_domains":["incident","api_reference"],"is_answerable":true,"ground_truth_answer":"...(선택)","_draft":false}
```
- `qid`: 고유 키 (qrels와 조인). 티어별 접두사 관례 — `d0xx` 단일 도메인 single-hop,
  `m0xx` 멀티 도메인, `h0xx` multi-hop(인접 청크 2~3개 종합, 멀티청크 qrels),
  `n0xx` near-miss unanswerable(도메인 안 주제지만 코퍼스에 답 없음),
  `c0xx` confusable(유사 문서 군집 속 타깃 청크 저격).
- `domain`: **과거 단일 라우터 정답 / 하위호환 필드**(`incident|manual|api_reference|meeting_note|planning` 또는 `null`).
  #86 피벗 이후 **운영 검색 기본은 soft** — 문서 단일 `domain`과 질의 `domains[]`(라우터 멀티)를 비교해 **점수 가산**(필터 아님).
  현재 질의 도메인 정답은 `router_domains`다. `domain`을 입력으로 쓰는 **hard/hybrid 필터는 비교 평가·후속 옵션**으로만 보존(#49).
- `gold_domains`: **정답 청크들이 실제로 걸친 도메인 집합**(선택). 장애 대응·온보딩처럼 근거가
  여러 도메인에 흩어진 질문은 `len>=2`(멀티 도메인). 생략 시 answerable이면 `[domain]`로 기본.
  `domain`(라우터 단일 픽)은 반드시 `gold_domains`에 포함돼야 한다(로더가 검증).
- `router_domains`: **질의를 라우터가 분류해야 하는 순서 있는 도메인 정답**(선택, #61). 순서=우선순위,
  answerable은 1~2개·중복 금지·`Domain` enum만. **`gold_domains`(정답 *문서*가 걸친 도메인)와 의미가 다르다**
  — 이쪽은 *질의 의도*다. 둘을 같은 값으로 재사용 금지(우연히 같을 수는 있음). 로더는 **출처**(`router_domains_source`)를
  구분한다: 명시값=`explicit`, 필드 없으면 `[domain]`(domain 없으면 빈 무필터)=`fallback`, unanswerable=`none`(`[]`).
  명시적 `[]`(answerable)은 거부(빈 정답=결함). **공식 라우터 지표는 `explicit`만 사용**하고 `fallback`은 제외하므로,
  검수 전 단일 fallback이 멀티 평가를 오염시키지 않는다. **최종 평가 전 사람 검수 필수**(아래 멀티 도메인 라우터 평가 참고).
- `is_answerable`: answerability 정확도 측정용. 범위 밖(답변 불가) 질문 일부 포함.
- `ground_truth_answer`: 선택. RAGAS LLM-judge(#C) 전용 — 검색 평가(#A)는 미사용.
- `_draft`: 부트스트랩 초안 표시. **팀 검수 후 제거**.

> **세 도메인 필드 역할 분리** — `domain`(과거 단일 라우터 정답·하위호환, hard/hybrid 필터 입력) ·
> `gold_domains`(정답 *문서*가 걸친 도메인 = **relevance 평가 메타데이터**, #65 분석용 — 운영 검색엔 미사용) ·
> `router_domains`(*질의* 의도 도메인 정답 = 현재 멀티도메인 라우터 평가, #61). 셋은 의미가 다르며
> 재사용 금지. 운영 soft 가산은 **문서 payload `domain` ∈ 질의 `domains[]`**로 계산하며 `gold_domains`는 쓰지 않는다.

### `qrels.jsonl`
```json
{"qid":"q001","relevant_chunk_ids":["<page_id>_003","<page_id>_004"]}
```
- unanswerable 질문이면 `[]`.
- 멀티 도메인 질문은 정답 청크가 2~3개 도메인의 서로 다른 페이지에 걸친다.

## 구축 워크플로우

1. **초안 부트스트랩** — `python scripts/bootstrap_golden.py --mode <single|multi-hop|near-miss|confusable>`
   (Qdrant 색인분에서 샘플링해 LLM으로 질문 생성, `_draft:true`로 모드별 `*.{mode}.draft.jsonl` 출력)
   - `multi-hop`: 같은 페이지 인접 청크 2~3개 종합 질문 → 멀티청크 qrels (Recall이 Hit Rate와 분리)
   - `near-miss`: 도메인 안 주제지만 코퍼스가 답하지 않는 unanswerable — '점심 메뉴'류보다 answerability 변별력 높음
   - `confusable`: 벡터 이웃(타 페이지 유사 청크) 많은 타깃 저격 질문 — 유사 문서 변별 측정
2. **pooling 검수 보조** — `python scripts/pool_candidates.py --queries <draft>` 로 질문별 rerank top-10
   후보를 뽑아, **정답인데 qrels에 없는 chunk_id를 보완**한다(라벨 누락 = 체계적 과소평가 원인).
   특히 중복 문서·튜토리얼/레퍼런스 중복이 많은 전체 코퍼스(864p)에서 필수.
3. **팀 검수** — 질문 자연스러움·관련 chunk_id 정확성 확인, paraphrase로 다양화(문구 베끼기 누수 방지),
   near-miss는 "정말 코퍼스에 답이 없는지" 교차 확인 후 `_draft` 제거.
4. **확정** — 도메인·티어 균형(목표 ~110문항), qid 충돌 확인, τ 재보정(`calibrate_answerability.py`) 후
   `--write-baseline` 재고정.

### 현재 구성 (118문항, #81 확장 병합)

| 티어 | 수 | 비고 |
|---|---|---|
| 단일 도메인 single-hop (`d0xx`) | 51 | 5종 도메인 (meeting_note·planning 포함, #75 전체 적재 이후) |
| 멀티 도메인 (`m0xx`) | 9 | 정답이 2~3개 도메인에 걸침 — #65 측정용 |
| multi-hop (`h0xx`) | 10 | 같은 페이지 인접 청크 2개 종합 — 멀티청크 qrels |
| confusable (`c0xx`) | 12 | 유사 문서 군집 속 타깃 청크 저격 — 리랭커 변별 측정 |
| 범위 밖 unanswerable (`d0xx`) | 16 | "점심 메뉴"류 — Router 차단 측정 |
| near-miss unanswerable (`n0xx`) | 20 | 도메인 내 주제지만 코퍼스에 답 없음 — τ 변별 측정 |

> 골든셋 구축 당시(53페이지 부분 색인)에는 `api_reference`·`manual`·`incident` 3종 도메인만
> 존재해 `meeting_note`·`planning`을 다루지 않았다. #75 전체 적재(864페이지, 5,731청크) 이후
> 코퍼스에 5종 도메인이 모두 존재하므로, 골든셋 확장 시 두 도메인 문항을 추가해야 한다.
> 멀티 도메인 문항은 현재 3종 조합으로 구성.

> **baseline 변동 이력** — #75 전체 적재로 코퍼스가 614→5,731청크(9.3배)가 되며 rerank
> Hit Rate@5 0.911→0.533 등 전 지표가 하락했다. 이는 회귀가 아니라 **실제 난이도 반영**이다
> (작은 코퍼스 점수는 distractor 부재로 과대평가). baseline.json의 `config.corpus`
> (컬렉션·포인트 수)로 측정 당시 코퍼스 상태를 식별한다.

### 멀티 도메인 측정
```bash
python scripts/eval_domain_filter.py --structural-only  # gold_domains 기반 구조 분석(오프라인)
python scripts/eval_domain_filter.py --mode dense       # filter ON/OFF recall 격차 실측(#65)
```

### 멀티 도메인 라우터 평가 (#61)

질의 멀티도메인 라우터의 분류 품질을 측정한다. **정답 = `router_domains`(사람 검수)**,
**예측 = 라우터 캐시의 `raw_predicted_domains`(분류·calibration) + `predicted_domains`(게이트 이후 운영 결과)**.
정답과 두 예측을 raw·effective 두 관점으로 비교한다.

```bash
# 1) 예측 캐시 생성 + 지표 리포트 (라우터 LLM 1회/질문, 신선 캐시는 재사용, Qdrant 불필요)
python scripts/eval_router_domains.py
python scripts/eval_router_domains.py --build-cache    # 예측 캐시만 생성/갱신(리포트 생략)
python scripts/eval_router_domains.py --report         # LLM 없이 캐시만으로 리포트
python scripts/eval_router_domains.py --report --write-result  # 캐시 기반 + baseline JSON 결정론적 저장

# 2) 사람 검수표 초안 생성 (캐시 있으면 예측을 제안값으로 채움)
python scripts/draft_router_domains.py
python scripts/draft_router_domains.py --blind   # 중요 문항 제안을 가려 독립 라벨링(앵커링 완화)
#    → data/eval/reviews/router_domains_review.jsonl (review_status: pending)
```

- **예측 캐시** `.cache/onramp-eval/router_predictions.jsonl` — **gitignore**(실행 환경·모델별 비결정 산출물).
  **오프라인 평가 전용 — production 런타임(route_node)은 읽지도 쓰지도 않는다.** 평가 스크립트를 수동 실행할
  때만 생성되고, 사용자 질문마다 누적되지 않으며(현재 골든셋 기준 전체를 덮어씀, append 아님), 운영 컨테이너에서
  평가를 돌리지 않으면 디스크가 늘지 않는다(Helm PVC·Redis 불필요). stale 키 = `qid + query_sha + requested_model
  + effective_provider + llm_provider + default_model + prompt_sha + schema_version`. `commit_sha`·`created_at`는
  재현 메타로만 저장. **query 평문 미저장**. 프롬프트·모델·계약이 바뀌면 해당 qid만 자동 재예측한다.
  레코드는 `raw_predicted_domains`(게이팅 전)와 `predicted_domains`(게이팅 후)를 **둘 다** 저장 —
  "분류가 틀린 것"과 "도메인은 맞지만 저신뢰로 비워진 것"을 구분하기 위함.
- **검수표** `reviews/router_domains_review.jsonl` — **Git 추적**(합성 질문이라 평문 OK). 행마다
  `query_sha`·`suggestion_source`(router_prediction|none)·`proposed_router_domains`(제안)·`reviewed_router_domains`(사람)·
  `review_status`·`reviewer`·`reviewed_at`. **자동 제안을 그대로 정답화하지 않는다** — 사람이 `reviewed_*`를
  채운(approved/edited) 행만 `queries.jsonl`의 `router_domains`로 반영한다(자기 정답화 방지). 우선 검수: 멀티(`m0xx`)·confusable(`c0xx`).
  재실행 시 검수 결과는 qid로 보존하되 **`query_sha`가 일치할 때만** — 질문 문구가 바뀌면 옛 검수를 `pending`으로
  초기화해 재검수를 강제한다(질문 변경에도 옛 라벨이 따라붙는 stale 검수 방지).
- 지표(answerable ∧ **explicit** `router_domains`)는 **raw·effective 두 관점**으로 분리해 낸다:
  - `raw_classification_and_calibration`: 게이팅 **전**(`raw_predicted_domains`) → 라우터 분류 능력 + **calibration(ECE)**.
  - `effective_after_gate`: 게이팅 **후**(`predicted_domains`) → 운영 결과. calibration(ECE·confidence_bins)은 **빼서** 표시한다
    (게이팅 후 빈 예측을 오답 처리하면 calibration이 왜곡되므로 raw 기준만 유효).
  - 공통: primary accuracy · exact ordered/set match · micro P/R/F1 · macro-label P/R/F1 · 도메인별 P/R/F1 ·
    secondary precision/과다·미예측률 · parse 실패 수 · low-confidence empty 수 · UNANSWERABLE 차단 정확도(별도).
  - ECE(raw, primary 기준): parse 실패·confidence 없음은 제외하고 **제외 수 보고**. **모든 분모 0은 0 또는 N/A로 명시.**

- **baseline 결과** `results/router_domains_baseline.json` — **Git 추적**. `--write-result`로 **결정론적 생성**.
  재현 메타(골든 SHA·캐시 stale 키 필드 requested_model/effective_provider/llm_provider/default_model·prompt_sha·
  schema_version·confidence threshold·commit) + raw/effective 지표 + 도메인별 P/R/F1 + **UNANSWERABLE 차단을 near-miss/사외 분리** 집계.
  완전 재현은 같은 조건으로 `--build-cache` 재생성 필요(캐시는 gitignore·LLM 비결정).

> 현재 `queries.jsonl`의 `router_domains`는 **사람 검수 완료(81건 explicit, 2026-06-12)**. 검수표는 동일 배치 반영이라
> `reviewed_at`이 같다(제안 수용 21=approved, 변경·blind 독립작성 60=edited). 중복 질문 4쌍 제거 후 측정. baseline은 위 결과 파일 참조.
> (검수 전엔 fallback이라 지표가 안 나오고, `explicit`이 0건이면 `eval_router_domains.py`는 지표 대신 검수 절차를 안내한다.)

## 사용

### 검색 평가 (#A — 결정론, CI 게이트)
```bash
make eval                                 # dense vs rerank 점수표
python scripts/eval_retrieval.py --write-baseline   # baseline.json 고정
python scripts/eval_retrieval.py --gate   # baseline 대비 회귀 시 exit 1
```

### 생성 평가 (#C — RAGAS LLM-judge, 비차단·nightly)
```bash
uv pip install -e ".[eval]"               # ragas optional 의존성 설치
make eval-gen                             # Faithfulness / Answer Relevancy 점수표
python scripts/eval_generation.py --limit 10 --write-report   # 소규모 + gen_report.json 기록
```
- **reference-free**: `ground_truth_answer` 없이 Faithfulness(환각)·Answer Relevancy(관련성)만 측정.
- 골든셋의 `is_answerable:true`만 대상. 답변 보류·무근거 샘플은 자동 제외.
- LLM-judge는 실행마다 변동 → **회귀 게이트로 쓰지 않음**(nightly·수동, 추세 기록).

### 생성 평가 — reference 기반 지표 (#67, GT 답변 필요)
```bash
# 1) GT 답변 초안 부트스트랩 (정답 chunk 근거로 LLM이 모범답안 생성 → *.draft.jsonl, gitignore)
python scripts/bootstrap_gt_answers.py --limit 10
#    → 팀 검수·paraphrase 후 queries.jsonl의 ground_truth_answer로 병합
# 2) reference 지표 포함 채점 (GT 있는 문항만)
python scripts/eval_generation.py --with-reference --limit 10
```
- `--with-reference`: **FactualCorrectness(정답성)·SemanticSimilarity(의미유사도)** 추가 채점.
- GT(`ground_truth_answer`) 없는 문항은 reference 지표에서 자동 제외(reference-free는 그대로).

> `qid`는 검수 중 일부 문항을 제외하면 **비연속**일 수 있다(예: d010 제외). 로더는 queries↔qrels가 qid로
> 일치하기만 하면 연속성을 요구하지 않는다.
