"""Answer Agent 노드 — 5요소 구조화 답변 생성 + Answerability Status 판정.

P0: 결정론 floor(문서 0건·LLM 실패·파싱 실패) + LLM 자기판정 + 인용 guard 하이브리드.
P1: Trust(Evidence Confidence)가 evidence_score·gate를 넘기면 decide_answerability가 점수/게이트로 분기.
async 노드이므로 그래프는 ainvoke로 실행한다.
"""

from __future__ import annotations

import logging

from app.agents.answer.answerability import (
    NO_SOURCE_REASON,
    decide_answerability,
    reason_for,
)
from app.agents.answer.formatter import format_answer
from app.agents.answer.prompts import ANSWER_SYSTEM_PROMPT
from app.agents.state import AgentState, AnswerabilityStatus, FiveElements, SourceDocument
from app.services.llm_selector import call_llm

logger = logging.getLogger(__name__)

# 보류 상태 — 5요소를 비우고 안내 메시지만 제공
_HOLD_STATUSES = {
    AnswerabilityStatus.NOT_ENOUGH_EVIDENCE,
    AnswerabilityStatus.CONFLICTING_EVIDENCE,
    AnswerabilityStatus.OUTDATED_EVIDENCE,
}


def _result(
    answer: FiveElements,
    sources: list[SourceDocument],
    status: AnswerabilityStatus,
    reason: str,
    error: str = "",
) -> dict:
    out: dict = {
        "answer": answer,
        "sources": sources,
        "answerability_status": status,
        "answerability_reason": reason,
        "agent_trace": ["answer"],
    }
    if error:
        out["error"] = error
    return out


def _build_context(documents: list[SourceDocument]) -> str:
    return "\n\n".join(f"[{i}] title: {doc.title}\ncontent: {doc.content_snippet}" for i, doc in enumerate(documents))


async def answer_node(state: AgentState) -> dict:
    """문서 근거로 5요소 답변을 생성하고 Answerability Status를 판정한다."""
    documents = state.get("documents", [])
    query = state.get("refined_query") or state.get("query", "")

    # 결정론 floor: 문서 0건 → 보류 (LLM 호출 안 함)
    if not documents:
        status = decide_answerability(documents)
        return _result(FiveElements(), [], status, reason_for(status))

    user_prompt = f"문서 컨텍스트:\n{_build_context(documents)}\n\n질문: {query}"
    try:
        raw = await call_llm(ANSWER_SYSTEM_PROMPT, user_prompt, model=state.get("model", ""), json_mode=True)
    except Exception as exc:  # LLM 호출 실패 → 보류
        logger.warning("Answer LLM 호출 실패 — NOT_ENOUGH fallback", exc_info=True)
        status = AnswerabilityStatus.NOT_ENOUGH_EVIDENCE
        return _result(FiveElements(), [], status, reason_for(status), error=str(exc))

    five, sources, llm_status, parse_ok = format_answer(raw, documents)
    if not parse_ok:  # JSON/스키마 파싱 실패 → 보류
        logger.warning("Answer 응답 파싱 실패 — NOT_ENOUGH fallback")
        return _result(FiveElements(), [], AnswerabilityStatus.NOT_ENOUGH_EVIDENCE, "답변 생성 실패 (파싱 오류)")

    # P1: Trust 게이트(충돌/만료/민감차단)를 먼저 적용하고, 그 외엔 P0 LLM 자기판정.
    #   evidence_score는 v1에서 의도적으로 미사용 — trust.overall(가중평균 ~0.x 스케일)이
    #   answerability 임계값(0.80/0.60)과 스케일이 달라 별도 보정 전까지 연결하지 않는다(track-B).
    #   게이트 신호(gate_flags)는 스케일 무관 불리언이라 안전하게 우선 적용한다.
    gate = state.get("gate_flags")
    status = decide_answerability(documents, gate=gate, llm_status=llm_status)

    # 인용 guard: ANSWERABLE인데 인용 출처 0건 → PARTIALLY로 강등
    if status == AnswerabilityStatus.ANSWERABLE and not sources:
        status, reason = AnswerabilityStatus.PARTIALLY_ANSWERABLE, NO_SOURCE_REASON
    else:
        reason = reason_for(status)

    # 상태별 응답 처리: 보류 상태는 5요소 비움.
    # 단 CONFLICTING/OUTDATED는 "왜 보류됐는지" 근거 문서를 보여줘야 하므로 sources는 유지(P1).
    if status in _HOLD_STATUSES:
        held_sources = [] if status == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE else sources
        return _result(FiveElements(), held_sources, status, reason)
    return _result(five, sources, status, reason)
