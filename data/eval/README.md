# 검색 평가 골든셋 (`data/eval/`)

우리 코퍼스(Qdrant 색인분) 위에서 검색 품질을 **결정론적으로** 측정하기 위한 정답지.
관련성 단위는 **chunk_id** (Qdrant payload의 `chunk_id`, 형식 `{page_id}_{idx:03d}`).

## 파일

| 파일 | 내용 |
|---|---|
| `queries.jsonl` | 질문 + 메타 (1줄=1질문) |
| `qrels.jsonl` | 질문별 정답 chunk_id 라벨 (1줄=1질문) |
| `baseline.json` | 검색 베이스라인 수치 (회귀 기준, CLI가 생성) |
| `gen_report.json` | 생성 평가(RAGAS) 최근 리포트 — **gitignore**(LLM-judge 비결정 → 로컬/CI 아티팩트로만) |

### `queries.jsonl`
```json
{"qid":"q001","query":"...","domain":"incident","gold_domains":["incident","api_reference"],"is_answerable":true,"ground_truth_answer":"...(선택)","_draft":false}
```
- `qid`: 고유 키 (qrels와 조인). 티어별 접두사 관례 — `d0xx` 단일 도메인 single-hop,
  `m0xx` 멀티 도메인, `h0xx` multi-hop(인접 청크 2~3개 종합, 멀티청크 qrels),
  `n0xx` near-miss unanswerable(도메인 안 주제지만 코퍼스에 답 없음),
  `c0xx` confusable(유사 문서 군집 속 타깃 청크 저격).
- `domain`: **라우터가 고를 단일 도메인** = 프로덕션 하드 필터 입력. `incident|manual|api_reference|meeting_note|planning` 또는 `null`(무필터).
- `gold_domains`: **정답 청크들이 실제로 걸친 도메인 집합**(선택). 장애 대응·온보딩처럼 근거가
  여러 도메인에 흩어진 질문은 `len>=2`(멀티 도메인). 생략 시 answerable이면 `[domain]`로 기본.
  `domain`(라우터 단일 픽)은 반드시 `gold_domains`에 포함돼야 한다(로더가 검증).
- `is_answerable`: answerability 정확도 측정용. 범위 밖(답변 불가) 질문 일부 포함.
- `ground_truth_answer`: 선택. RAGAS LLM-judge(#C) 전용 — 검색 평가(#A)는 미사용.
- `_draft`: 부트스트랩 초안 표시. **팀 검수 후 제거**.

> **`domain` vs `gold_domains` (역할 분리, IR 골든셋 모범사례)** — `domain`은 질문의 *의도 facet*
> (라우터 단일 픽, #65의 하드 필터가 쓰는 값)이고, `gold_domains`는 *relevance judgment의 도메인
> 커버리지*다. 둘을 분리해야 "단일 도메인 필터가 멀티 도메인 정답을 배제한다"(#65)를 골든셋 파일만으로
> 측정할 수 있다.

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

### 현재 구성 (58문항)

| 티어 | 수 | 비고 |
|---|---|---|
| 단일 도메인 answerable | 36 | manual 12 / api_reference 12 / incident 12 |
| 멀티 도메인 answerable (`m0xx`) | 9 | 정답이 2~3개 도메인에 걸침 — #65 측정용 |
| unanswerable | 13 | answerability 정확도 측정용 |

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
