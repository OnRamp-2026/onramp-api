# RAG 평가 — KDT 강의노트 정리 (#C 착수용 레퍼런스)

> 출처: KDT 생성형AI 「3. RAG Pipeline 설계 및 구축」 — **4. RAG 평가** (p.122–143)
> 목적: #C(생성 평가/RAGAS) 설계 전, 강의노트의 평가 프레임을 정리하고 OnRamp 현황에 매핑.

---

## 0. 전체 프레임 — 2개의 축

강의노트는 RAG 평가를 **두 질문**으로 나눈다.

| 축 | 질문 | 구성 |
|---|---|---|
| **What to measure** (무엇을 잴까) | 평가 *지표* | ① Heuristic(전통 지표) · ② LLM-as-a-Judge |
| **How to evaluate using LLM** (어떻게 잴까) | 평가 *도구/프레임워크* | ③ RAGAS · ④ LangSmith |

핵심 메시지: **텍스트 생성 평가는 Heuristic → LLM 기반으로 진화 중이며, 평가 목적에 맞는 지표를 선택**해야 한다.

---

## 1. Heuristic 지표 (단어/표면 유사도, LLM 불필요)

참조 답변(reference)과 생성 답변의 **표면적 일치**를 본다. 결정론·저비용·빠름.

### (1) ROUGE — Recall 중심
- 원래 **요약(Gisting) 품질** 평가용. n-gram을 얼마나 공유하는지.
- **ROUGE-1**(unigram, 단어 일치도) / **ROUGE-2**(bigram, 작은 문맥) / **ROUGE-L**(LCS, 전반적 유사성·순서 유지)
- 강점: 재현율 기반 → 요약 평가에 강함, 키워드 포함 여부
- 단점: 의미적 유사성 반영 불가, 단순 단어 매칭 의존

### (2) BLEU — Precision 중심
- 주로 **기계번역** 평가. n-gram precision(1~4gram) + **Brevity Penalty**(짧은 문장 패널티) + 기하평균
- 예: 한 단어라도 안 맞으면 해당 n-gram precision=0 → BLEU 0 (의미 유사도 미반영)
- 강점: 단순·빠른 정량 평가 / 단점: 의미·재현율 미반영, 짧은 문장 과대평가

### (3) METEOR — Precision+Recall 조화 + 의미 매칭
- BLEU 한계 보완. **정확 매칭 + 어간(run/running) + 동의어(quick/fast) + 의역** 매칭 고려
- 정밀도·재현율의 **조화평균(F)** × (1 − 순서 패널티)
- 강점: 의미적 유사성 일부 반영, 순서 고려 / 단점: 어간·동의어 사전이 **언어 종속**(한국어 구현 제약)

> **OnRamp 시사점**: 3개 모두 *참조 답변(ground-truth answer)* 필요 + *한국어 표면매칭 한계*. #A에서 BEIR/표면지표를 1차 메트릭에서 제외한 판단과 일치. #C에서도 보조/스모크용으로만.

---

## 2. LLM-as-a-Judge (의미·복합 기준, LLM 사용)

### (1) SemScore — 의미적 텍스트 유사도(STS)
- 문장 임베딩 → **cosine similarity**로 참조-생성 답변의 의미 유사도 정량화
- 단순 n-gram을 넘어 문맥/의미 포착(BERT/RoBERTa류). Human 평가와 높은 상관
- 단점: 점수 해석이 직관적이지 않음, 임베딩 모델 품질 의존

### (2) LLM 직접 평가 (프롬프트 기반)
- LLM에 **평가자 역할** 부여 → 정확성/충실성/관련성/유창성 등으로 채점(점수+설명)
- 프롬프트 구성: System(평가자 역할) + User(질문+문서+응답+기준)
- **주의사항(편향)**: 일관성 부족, 기준 모호성, 모델 편향, 과대 관대화, **자기 모델 평가 편향**(GPT가 GPT를 후하게)

### (3) G-Eval — 표준화된 LLM 평가
- **CoT + Form-filling** 방식으로 LLM 유도 → AMBIGUITY 제거
- 명확한 평가 항목 제시: **Correctness / Faithfulness / Relevance / Consistency / Clarity / Fluency / Conciseness**
- 편차 최소화: 출력 양식 통제(Form-filling), 반복 수행 또는 **Multi-LLM 교차 평가**

> **OnRamp 시사점**: 편향 관리(특히 자기평가 편향, 기준 명확화, 교차/반복)가 #C LLM-judge 설계의 핵심 체크리스트. nightly/비차단으로 두기로 한 우리 결정과 부합.

---

## 3. RAGAS — Retrieval-Augmented Generation Assessment Score ⭐

검색(Retrieval)과 생성(Generation)을 **각각 평가 후 통합 점수** 제공.

### A. Test Dataset 생성 (Synthetic)
- 구성요소: **Question · Contexts(관련문서) · Ground_Truth(참조답변) · Metadata**
- `TestsetGenerator`로 합성: 분포 `simple / reasoning / multi_context / conditional`
```python
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings, doc_store)
distributions = {simple:0.4, reasoning:0.2, multi_context:0.2, conditional:0.2}
test_set = generator.generate_with_langchain_docs(documents=docs, test_size=10, distributions=distributions)
```

### B. 답변 평가 지표

**Retrieval Score (검색 평가)**
| 지표 | 정의 | 입력 관계 |
|---|---|---|
| **Context Precision** (정밀도) | 검색된 문서 중 상위 결과가 질의와 얼마나 관련 있나 | Query ↔ Retrieval |
| **Context Recall** (재현율) | 정답 문장이 검색 문서에 얼마나 담겼나(누락 여부) | Retrieval ↔ Answer(GT) |

**Generation Score (생성 평가)**
| 지표 | 정의 | 입력 관계 |
|---|---|---|
| **Faithfulness** (사실성) | 응답이 주어진 문서 기반인가 (Hallucination 방지) | Context+Retrieval ↔ Answer |
| **Answer Relevancy** (응답 관련성) | 응답에서 잠재 질문 생성 → 원 질문과 임베딩 유사도 | Query ↔ Answer |
| **Context Relevancy** (문맥 관련성) | 답변이 실제 검색 문서 내용과 연결됐나 | Retrieval ↔ Answer |
| **Conciseness** (간결성) | 불필요한 중복 없이 명확한가 | Answer |
| **Fluency** (유창성) | 문장구조·표현·문법이 자연스러운가 | Answer |

### 워크된 예시 (p.136)
질문 "인공지능이 의료 분야에 미치는 영향?" / 5문서 검색:
- Context Precision = 3/5 = **0.6** (관련 3건)
- Context Recall = 3/3 = **1.0**
- Faithfulness = 3/3 = **1.0** (3개 주장 모두 문서 근거)
- Answer Relevancy = **0.92** (cos 평균)
- Context Relevancy = **0.85** / Conciseness = **0.95** / Fluency = **1.0**
- **RAGAS Score = 0.90** (weighted average, 항목 동등 가중)

---

## 4. LangSmith — 평가 전과정 관리 도구

- **Test Dataset 생성·관리**: RAGAS/HuggingFace dataset 관리, 버전관리, 평가결과 관리
- **답변 평가**: LLM-as-a-Judge(질문-응답 일관성·논리·관련성) / Answer Relevance(GT↔Answer cosine·euclidean) / **Hallucination 평가**(Retrieval↔Answer, Groundedness Evaluator)
- **Trace & Feedback**: 평가점수 + 추론경로(Chain/Agent trace) 시각화, 사용자 manual feedback 수집(online 포함)

> **OnRamp 시사점**: LangSmith는 우리 LangGraph trace와 자연스럽게 결합 가능(관측/온라인 피드백). 단, 외부 SaaS 의존 — 도입은 별도 결정사항.

---

## 5. OnRamp 매핑 — 우리가 가진 것 vs 강의노트 기준 채울 것

| 강의노트 항목 | OnRamp 현황 | #C 액션 |
|---|---|---|
| 검색 결정론 지표 (Hit/MRR/Recall/nDCG) | ✅ #A 완료 (chunk_id 단위, CI 게이트) | 유지 |
| RAGAS **Context Recall/Precision** | △ #A가 chunk_id qrels 보유 → **non-LLM Context Recall 재사용 가능** | IDBased/NonLLM Context Recall 우선(비-LLM·CI) |
| RAGAS **Faithfulness** | ❌ 없음 | LLM-judge, nightly/비차단 |
| RAGAS **Answer Relevancy** | ❌ 없음 | LLM-judge, nightly/비차단 |
| **Ground-truth answer** | ❌ 골든셋엔 `is_answerable`만, GT답변 없음 | #C에서 GT답변 컬럼 추가(부트스트랩+검수) |
| Synthetic Test Dataset | △ #A 골든셋 60건(수기검수) 보유 | 합성 생성보다 **기존 검수 골든셋 우선 재사용** |
| Heuristic(ROUGE/BLEU/METEOR) | ❌ | 한국어 한계 → 스모크/보조만 |
| LLM 편향 관리(G-Eval) | — | judge 프롬프트: 기준 명시 + Form-filling + 반복/교차 |
| LangSmith | ❌ | 옵션(외부 의존), trace 결합 시 검토 |

### #C 설계 시 확정 권고 (강의노트 근거)
1. **비-LLM 우선**: Context Recall/Precision은 #A의 chunk_id qrels로 **LLM 없이** 계산 → CI 게이트(결정론 유지).
2. **LLM-judge는 nightly·비차단**: Faithfulness/Answer Relevancy. 편향(자기평가/관대화) 관리 필수.
3. **GT 답변 필요**: Faithfulness/Answer Relevancy·SemScore 모두 참조답변 의존 → 골든셋에 `ground_truth_answer` 추가가 #C의 선결 작업.
4. **Heuristic는 후순위**: 한국어 표면매칭 한계로 메인 지표 아님(보조/스모크).

---

## 부록 — 강의노트 평가 페이지 인덱스
- p.123 평가 방법 전체 맵 / p.124–127 Heuristic(ROUGE·BLEU·METEOR·비교표)
- p.128–129 SemScore / p.130 LLM 직접평가 / p.131 G-Eval / p.132 지표 비교표
- p.133–136 RAGAS(구조·TestSet·답변평가·예시) / p.137 LangSmith
