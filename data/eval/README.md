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
{"qid":"q001","query":"...","domain":"incident","is_answerable":true,"ground_truth_answer":"...(선택)","_draft":false}
```
- `qid`: 고유 키 (qrels와 조인).
- `domain`: `incident|manual|api_reference|meeting_note|planning` 또는 `null`(무필터).
- `is_answerable`: answerability 정확도 측정용. 범위 밖(답변 불가) 질문 일부 포함.
- `ground_truth_answer`: 선택. RAGAS LLM-judge(#C) 전용 — 검색 평가(#A)는 미사용.
- `_draft`: 부트스트랩 초안 표시. **팀 검수 후 제거**.

### `qrels.jsonl`
```json
{"qid":"q001","relevant_chunk_ids":["<page_id>_003","<page_id>_004"]}
```
- unanswerable 질문이면 `[]`.

## 구축 워크플로우

1. **초안 부트스트랩** — `python scripts/bootstrap_golden.py`
   (Qdrant 색인분에서 chunk를 샘플링해 "그 chunk가 답이 되는 질문"을 LLM으로 생성, `_draft:true`로 출력)
2. **팀 검수** — 질문 자연스러움·관련 chunk_id 정확성 확인, paraphrase로 다양화(문구 베끼기 누수 방지), `_draft` 제거.
3. **확정** — 도메인 5종 균형, 30~50문항 권장. unanswerable 케이스 일부 포함.

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

> `qid`는 검수 중 일부 문항을 제외하면 **비연속**일 수 있다(예: d010 제외). 로더는 queries↔qrels가 qid로
> 일치하기만 하면 연속성을 요구하지 않는다.
