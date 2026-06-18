"""생성 평가 어댑터 — query를 우리 운영 그래프로 흘려 답변/문맥을 추출한다.

검색 어댑터(retrieval_adapter.py)와 달리, 평가용 경로를 따로 미러하지 않고
**실 그래프**(`compiled_graph`: router→retriever→trust→answer)를 그대로 실행한다.
→ Trust 재검색·인용 guard 등 운영 동작이 모두 반영돼 생성 품질을 충실히 평가한다.

LLM-judge(RAGAS)는 ragas_judge.py가 담당하고, 여기서는 judge 입력
(질문/답변/문맥)만 만든다. unanswerable·보류(answer 비어있음)는 호출측에서 제외한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter

from app.agents.graph import compiled_graph
from app.agents.state import FiveElements, SourceDocument
from app.config import Settings, get_settings
from app.services.llm_selector import usage_accumulator

logger = logging.getLogger(__name__)

# FiveElements → judge용 단일 텍스트 합성 시 사용할 (필드, 한글 라벨) 순서.
_ELEMENT_LABELS: tuple[tuple[str, str], ...] = (
    ("situation", "상황"),
    ("cause", "원인"),
    ("evidence", "근거"),
    ("solution", "해결"),
    ("infra_context", "인프라 맥락"),
)


@dataclass(frozen=True)
class GenerationResult:
    """생성 평가용 결과 — RAGAS SingleTurnSample 입력 재료."""

    query: str
    answer_text: str  # FiveElements 5필드를 라벨 붙여 합친 단일 응답
    retrieved_contexts: list[str] = field(default_factory=list)  # 검색된 문맥(content_snippet)
    answerability_status: str = ""
    n_docs: int = 0
    reference: str | None = None  # 골든 GT 답변(#67) — reference 기반 지표용. 없으면 reference-free만.
    # 운영 비용 계측 (#212 — child-only↔parent-expanded ablation 비교축). 그래프 전체 1질의 기준.
    prompt_tokens: int = 0  # 그래프 내 모든 LLM 호출의 input token 합 (router+answer+재검색 등)
    completion_tokens: int = 0  # output token 합
    total_tokens: int = 0  # input+output 합
    llm_calls: int = 0  # 그래프가 발화한 LLM 호출 수
    latency_s: float = 0.0  # 그래프 ainvoke wall-clock (초)
    rerank_fallback: bool = False  # 리랭커 폴백 여부 — strict 모드 검증·오염 감지(#212 §2-5)

    @property
    def is_evaluable(self) -> bool:
        """judge 대상 여부 — 답변 본문과 문맥이 모두 있어야 한다(보류/무근거 제외)."""
        return bool(self.answer_text.strip()) and bool(self.retrieved_contexts)

    @property
    def has_reference(self) -> bool:
        """reference 기반 지표(FactualCorrectness 등) 채점 가능 여부."""
        return bool(self.reference and self.reference.strip())


def flatten_answer(answer: FiveElements | None) -> str:
    """FiveElements를 '라벨: 내용' 줄들로 합친다. 빈 필드는 건너뛴다."""
    if answer is None:
        return ""
    lines = []
    for attr, label in _ELEMENT_LABELS:
        value = str(getattr(answer, attr, "") or "").strip()
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _contexts(documents: list[SourceDocument]) -> list[str]:
    return [d.content_snippet for d in documents if d.content_snippet]


async def generate_for_eval(
    query: str,
    *,
    domain: str | None = None,  # noqa: ARG001 - 라우터가 도메인을 직접 분류하므로 미사용(시그니처 호환용)
    model: str = "",
    reference: str | None = None,  # 골든 GT 답변(#67) — reference 기반 지표용
    settings: Settings | None = None,
) -> GenerationResult:
    """query를 실 그래프로 실행해 생성 평가용 결과를 만든다."""
    settings = settings or get_settings()
    # token usage는 그래프 내부 call_llm들이 누산기에 합산한다(Langfuse 무관). latency는 ainvoke wall-clock.
    with usage_accumulator() as usage:
        started = perf_counter()
        state = await compiled_graph.ainvoke(
            {
                "query": query,
                "model": model,
                "retry_count": 0,
                "max_retries": settings.trust_max_retries,
            }
        )
        latency_s = perf_counter() - started
    documents = state.get("documents", []) or []
    answer_text = flatten_answer(state.get("answer"))
    # AnswerabilityStatus는 StrEnum → str()이 곧 enum value("answerable" 등). None이면 빈 문자열.
    status = state.get("answerability_status")
    return GenerationResult(
        query=query,
        answer_text=answer_text,
        retrieved_contexts=_contexts(documents),
        answerability_status=str(status) if status else "",
        n_docs=len(documents),
        reference=reference,
        prompt_tokens=usage["input"],
        completion_tokens=usage["output"],
        total_tokens=usage["total"],
        llm_calls=usage["calls"],
        latency_s=latency_s,
        rerank_fallback=bool(state.get("rerank_fallback")),
    )
