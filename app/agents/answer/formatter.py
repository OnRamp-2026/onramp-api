"""Answer LLM 응답 파싱 → 5요소 + 출처 매핑."""

from __future__ import annotations

from pydantic import ValidationError

from app.agents.answer.schema import AnswerOutput, FreeformOutput
from app.agents.state import AnswerabilityStatus, FiveElements, SourceDocument


def format_freeform(
    llm_response: str, documents: list[SourceDocument]
) -> tuple[str, list[SourceDocument], AnswerabilityStatus, bool]:
    """Freeform LLM JSON → (answer_text, 인용 sources, LLM 자기판정 status, 파싱 성공 여부).

    파싱 실패 시 ("", [], NOT_ENOUGH_EVIDENCE, False)를 반환한다. (format_answer와 평행 계약)
    """
    try:
        out = FreeformOutput.model_validate_json(llm_response)
    except ValidationError:
        return "", [], AnswerabilityStatus.NOT_ENOUGH_EVIDENCE, False

    sources = [documents[i] for i in out.source_indices if 0 <= i < len(documents)]
    return out.answer_text, sources, out.answerability_status, True


def format_answer(
    llm_response: str, documents: list[SourceDocument]
) -> tuple[FiveElements, list[SourceDocument], AnswerabilityStatus, bool]:
    """LLM JSON → (5요소, 인용 sources, LLM 자기판정 status, 파싱 성공 여부).

    JSON/스키마 파싱 실패 시 (빈 5요소, [], NOT_ENOUGH_EVIDENCE, False)를 반환한다.
    """
    try:
        out = AnswerOutput.model_validate_json(llm_response)
    except ValidationError:
        return FiveElements(), [], AnswerabilityStatus.NOT_ENOUGH_EVIDENCE, False

    five = FiveElements(
        situation=out.situation,
        cause=out.cause,
        evidence=out.evidence,
        solution=out.solution,
        infra_context=out.infra_context,
    )
    sources = [documents[i] for i in out.source_indices if 0 <= i < len(documents)]
    return five, sources, out.answerability_status, True
