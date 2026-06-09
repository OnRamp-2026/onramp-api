"""RAGAS LLM-judge 래퍼 — 생성 답변을 reference-free / reference 기반 지표로 채점.

설계(#C, #67):
    - ragas는 **optional 의존성**(`pip install -e ".[eval]"`). 미설치 시에도 이 모듈 import는
      성공하고(lazy import), 실제 채점 시점에만 ragas를 들여온다. → prod/CI 비의존.
    - reference-free (GT 답변 불필요, #C):
        · Faithfulness     : 답변이 검색 문맥에 근거하는가 (환각 탐지)
        · ResponseRelevancy: 답변이 질문에 관련되는가 (구 AnswerRelevancy)
    - reference 기반 (골든 GT 답변 필요, #67 — with_reference=True일 때만):
        · FactualCorrectness : 답변이 참조답변과 사실 일치하는가 (정답성)
        · SemanticSimilarity : 답변↔참조 임베딩 유사도
      GT가 없는 샘플은 reference 지표에서 자동 제외(reference-free는 그대로 채점).
    - judge LLM/임베딩은 ragas 요구상 langchain OpenAI를 직접 사용한다(우리 call_llm/
      provider 추상화 우회 — eval 한정·OpenAI 가정). 비결정적이라 CI 게이트 금지(nightly·비차단).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.eval.generation_adapter import GenerationResult

logger = logging.getLogger(__name__)

# judge 기본 모델 — OpenAI 계열(ragas evaluator). settings.default_model이 gpt 계열이면 그걸 쓴다.
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


def ragas_available() -> bool:
    """ragas(+langchain-openai) import 가능 여부. CLI/테스트가 skip 판단에 사용."""
    try:
        import langchain_openai  # noqa: F401
        import ragas  # noqa: F401
    except ImportError:
        return False
    return True


def _round(v: float | None) -> float | None:
    return round(v, 4) if v is not None else None


@dataclass(frozen=True)
class GenerationScores:
    """생성 평가 매크로 평균. reference-free는 n_evaluated, reference 지표는 n_reference_evaluated 기준."""

    faithfulness: float | None
    answer_relevancy: float | None
    n_evaluated: int  # reference-free 두 지표 모두 성공해 평균에 기여한 샘플 수
    n_skipped: int  # reference-free 평균에 기여하지 못한 샘플 수 (보류·무근거 + 채점 실패)
    # reference 기반 (#67) — with_reference=True + GT 존재 시에만 채움
    factual_correctness: float | None = None
    semantic_similarity: float | None = None
    n_reference_evaluated: int = 0  # reference 두 지표 모두 성공한 샘플 수

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "faithfulness": _round(self.faithfulness),
            "answer_relevancy": _round(self.answer_relevancy),
            "n_evaluated": self.n_evaluated,
            "n_skipped": self.n_skipped,
            "factual_correctness": _round(self.factual_correctness),
            "semantic_similarity": _round(self.semantic_similarity),
            "n_reference_evaluated": self.n_reference_evaluated,
        }


def resolve_judge_model(settings: Settings) -> str:
    """실제 채점에 쓸 judge 모델. OpenAI 계열이면 default_model, 아니면 fallback.

    리포트(eval_generation.py)가 '실제 사용 모델'을 정확히 기록하도록 재사용한다.
    """
    model = (settings.default_model or "").strip()
    if model.startswith(("gpt", "o1", "o3", "o4")):
        return model
    return DEFAULT_JUDGE_MODEL


def _build_evaluator(settings: Settings):
    """ragas evaluator LLM/임베딩 래퍼를 만든다 (lazy import)."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from pydantic import SecretStr
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    api_key = SecretStr(settings.openai_api_key) if settings.openai_api_key else None
    llm = LangchainLLMWrapper(ChatOpenAI(model=resolve_judge_model(settings), temperature=0.0, api_key=api_key))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=settings.embedding_model, api_key=api_key))
    return llm, embeddings


def _mean(values: list[float]) -> float | None:
    """NaN/None을 제외한 평균. 유효값이 없으면 None."""
    valid = [v for v in values if v is not None and not math.isnan(v)]
    return sum(valid) / len(valid) if valid else None


async def _score_pair(metric_a, metric_b, samples):
    """샘플별로 두 지표를 함께 채점해 (a_scores, b_scores) 반환. 한쪽 실패 시 그 샘플 제외."""
    a_scores: list[float] = []
    b_scores: list[float] = []
    for r, sample in samples:
        try:
            a = await metric_a.single_turn_ascore(sample)
            b = await metric_b.single_turn_ascore(sample)
        except Exception:  # 개별 샘플 채점 실패 → 평균에서 제외(전체 중단 방지)
            logger.warning("RAGAS 채점 실패 (query=%.40s) — 해당 샘플 제외", r.query, exc_info=True)
            continue
        a_scores.append(a)
        b_scores.append(b)
    return a_scores, b_scores


async def score_generation(
    results: list[GenerationResult],
    *,
    with_reference: bool = False,
    settings: Settings | None = None,
) -> GenerationScores:
    """평가가능한 결과를 RAGAS로 채점·매크로 평균한다.

    reference-free(Faithfulness/ResponseRelevancy)는 항상, reference 기반
    (FactualCorrectness/SemanticSimilarity)은 with_reference=True + GT 존재 샘플에만 채점한다.
    ragas 미설치 시 ImportError. 호출측(CLI/테스트)이 ragas_available()로 먼저 거른다.
    """
    settings = settings or get_settings()
    evaluable = [r for r in results if r.is_evaluable]
    if not evaluable:
        return GenerationScores(None, None, 0, len(results))

    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness, ResponseRelevancy

    llm, embeddings = _build_evaluator(settings)

    # reference-free — 전체 evaluable
    rf_samples = [
        (r, SingleTurnSample(user_input=r.query, response=r.answer_text, retrieved_contexts=r.retrieved_contexts))
        for r in evaluable
    ]
    faith_scores, rel_scores = await _score_pair(
        Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=embeddings), rf_samples
    )
    n_evaluated = len(faith_scores)

    # reference 기반 (#67) — with_reference + GT 존재 샘플에만
    factual: float | None = None
    similarity: float | None = None
    n_ref = 0
    referable = [r for r in evaluable if r.has_reference] if with_reference else []
    if referable:
        from ragas.metrics import FactualCorrectness, SemanticSimilarity

        ref_samples = [
            (
                r,
                SingleTurnSample(
                    user_input=r.query,
                    response=r.answer_text,
                    retrieved_contexts=r.retrieved_contexts,
                    reference=r.reference,
                ),
            )
            for r in referable
        ]
        fc_scores, ss_scores = await _score_pair(
            FactualCorrectness(llm=llm), SemanticSimilarity(embeddings=embeddings), ref_samples
        )
        factual, similarity, n_ref = _mean(fc_scores), _mean(ss_scores), len(fc_scores)

    return GenerationScores(
        faithfulness=_mean(faith_scores),
        answer_relevancy=_mean(rel_scores),
        n_evaluated=n_evaluated,
        n_skipped=len(results) - n_evaluated,  # 보류·무근거 + 채점 실패 = 평균 미기여 전체
        factual_correctness=factual,
        semantic_similarity=similarity,
        n_reference_evaluated=n_ref,
    )
