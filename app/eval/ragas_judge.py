"""RAGAS LLM-judge 래퍼 — 생성 답변을 Faithfulness / Answer Relevancy로 채점.

설계(계획 #C):
    - ragas는 **optional 의존성**(`pip install -e ".[eval]"`). 미설치 시에도 이 모듈 import는
      성공하고(lazy import), 실제 채점 시점에만 ragas를 들여온다. → prod/CI 비의존.
    - reference-free 지표만 사용 → 골든셋 ground_truth_answer(참조답변) 불필요.
        · Faithfulness     : 답변이 검색 문맥에 근거하는가 (환각 탐지)
        · ResponseRelevancy: 답변이 질문에 관련되는가 (구 AnswerRelevancy)
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


@dataclass(frozen=True)
class GenerationScores:
    """생성 평가 매크로 평균. n_evaluated + n_skipped = 전체 입력 수."""

    faithfulness: float | None
    answer_relevancy: float | None
    n_evaluated: int  # 두 지표 모두 채점 성공해 평균에 기여한 샘플 수
    n_skipped: int  # 평균에 기여하지 못한 샘플 수 (보류·무근거 + 채점 실패)

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "faithfulness": round(self.faithfulness, 4) if self.faithfulness is not None else None,
            "answer_relevancy": round(self.answer_relevancy, 4) if self.answer_relevancy is not None else None,
            "n_evaluated": self.n_evaluated,
            "n_skipped": self.n_skipped,
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


async def score_generation(
    results: list[GenerationResult],
    *,
    settings: Settings | None = None,
) -> GenerationScores:
    """평가가능한 결과를 RAGAS Faithfulness/ResponseRelevancy로 채점·매크로 평균한다.

    ragas 미설치 시 ImportError. 호출측(CLI/테스트)이 ragas_available()로 먼저 거른다.
    """
    settings = settings or get_settings()
    evaluable = [r for r in results if r.is_evaluable]
    n_skipped = len(results) - len(evaluable)
    if not evaluable:
        return GenerationScores(None, None, 0, n_skipped)

    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness, ResponseRelevancy

    llm, embeddings = _build_evaluator(settings)
    faithfulness = Faithfulness(llm=llm)
    relevancy = ResponseRelevancy(llm=llm, embeddings=embeddings)

    faith_scores: list[float] = []
    rel_scores: list[float] = []
    for r in evaluable:
        sample = SingleTurnSample(
            user_input=r.query,
            response=r.answer_text,
            retrieved_contexts=r.retrieved_contexts,
        )
        try:
            # 두 지표를 모두 채점한 뒤에 함께 기록 — 한쪽만 성공해 표본이 어긋나는 것을 방지
            faith = await faithfulness.single_turn_ascore(sample)
            rel = await relevancy.single_turn_ascore(sample)
        except Exception:  # 개별 샘플 채점 실패 → 평균에서 제외(전체 중단 방지)
            logger.warning("RAGAS 채점 실패 (query=%.40s) — 해당 샘플 제외", r.query, exc_info=True)
            continue
        faith_scores.append(faith)
        rel_scores.append(rel)

    n_evaluated = len(faith_scores)  # 두 지표 모두 성공한 샘플 수(평균 기여)
    return GenerationScores(
        faithfulness=_mean(faith_scores),
        answer_relevancy=_mean(rel_scores),
        n_evaluated=n_evaluated,
        n_skipped=len(results) - n_evaluated,  # 보류·무근거 + 채점 실패 = 평균 미기여 전체
    )
