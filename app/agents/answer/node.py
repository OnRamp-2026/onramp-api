"""Answer Agent 노드 — 조건부 포맷(구조화/freeform) 답변 생성 + Answerability Status 판정.

포맷은 **사용자 의도(라우터 domains)** 로만 결정한다(#191): incident면 5요소 구조화, 그 외는 freeform 산문.
근거 품질(answerability)·grounding·sources·보류는 **두 포맷 공통**이며 포맷과 무관하게 동일 로직을 재사용한다.

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
from app.agents.answer.formatter import format_answer, format_freeform
from app.agents.answer.prompts import ANSWER_SYSTEM_PROMPT, FREEFORM_SYSTEM_PROMPT
from app.agents.state import AgentState, AnswerabilityStatus, Domain, FiveElements, SourceDocument
from app.config import get_settings
from app.services.llm_selector import call_llm

logger = logging.getLogger(__name__)

# 보류 상태 — 본문(5요소/answer_text)을 비우고 안내 메시지만 제공
_HOLD_STATUSES = {
    AnswerabilityStatus.NOT_ENOUGH_EVIDENCE,
    AnswerabilityStatus.CONFLICTING_EVIDENCE,
    AnswerabilityStatus.OUTDATED_EVIDENCE,
}


def _decide_answer_format(domains: list[Domain], structured_domains: set[str]) -> str:
    """라우터 domains가 structured 집합과 교집합이면 'structured', 아니면 'freeform' (#191).

    포맷은 사용자 의도(라우터)로만 결정한다 — 검색 근거 도메인은 포맷에 쓰지 않는다(answerability로 처리).
    domains가 비었거나(라우터 애매) 교집합이 없으면 freeform(안전한 기본).
    """
    return "structured" if any(d in structured_domains for d in domains) else "freeform"


def _result(
    *,
    answer_format: str,
    answer: FiveElements,
    answer_text: str,
    sources: list[SourceDocument],
    status: AnswerabilityStatus,
    reason: str,
    error: str = "",
) -> dict:
    out: dict = {
        "answer": answer,
        "answer_text": answer_text,
        "answer_format": answer_format,
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
    """문서 근거로 답변을 생성하고 Answerability Status를 판정한다 (포맷은 라우터 domains 기준)."""
    documents = state.get("documents", [])
    query = state.get("refined_query") or state.get("query", "")
    answer_format = _decide_answer_format(state.get("domains") or [], get_settings().structured_answer_domains)
    is_structured = answer_format == "structured"

    # 결정론 floor: 문서 0건 → 보류 (LLM 호출 안 함)
    if not documents:
        status = decide_answerability(documents)
        return _result(
            answer_format=answer_format,
            answer=FiveElements(),
            answer_text="",
            sources=[],
            status=status,
            reason=reason_for(status),
        )

    system_prompt = ANSWER_SYSTEM_PROMPT if is_structured else FREEFORM_SYSTEM_PROMPT
    user_prompt = f"문서 컨텍스트:\n{_build_context(documents)}\n\n질문: {query}"
    try:
        raw = await call_llm(system_prompt, user_prompt, model=state.get("model", ""), json_mode=True)
    except Exception as exc:  # LLM 호출 실패 → 보류
        logger.warning("Answer LLM 호출 실패 — NOT_ENOUGH fallback", exc_info=True)
        status = AnswerabilityStatus.NOT_ENOUGH_EVIDENCE
        return _result(
            answer_format=answer_format,
            answer=FiveElements(),
            answer_text="",
            sources=[],
            status=status,
            reason=reason_for(status),
            error=str(exc),
        )

    # 포맷별 파싱 — 본문만 다르고(5요소 vs answer_text) sources·llm_status·parse_ok 계약은 동일
    five = FiveElements()
    answer_text = ""
    if is_structured:
        five, sources, llm_status, parse_ok = format_answer(raw, documents)
    else:
        answer_text, sources, llm_status, parse_ok = format_freeform(raw, documents)
    if not parse_ok:  # JSON/스키마 파싱 실패 → 보류
        logger.warning("Answer 응답 파싱 실패 — NOT_ENOUGH fallback (format=%s)", answer_format)
        return _result(
            answer_format=answer_format,
            answer=FiveElements(),
            answer_text="",
            sources=[],
            status=AnswerabilityStatus.NOT_ENOUGH_EVIDENCE,
            reason="답변 생성 실패 (파싱 오류)",
        )

    # 인용 우선순위 (#108): Trust per-doc evidence가 높은 근거를 앞세운다 (안정 정렬 — 동률은 원 순서)
    sources = sorted(sources, key=lambda d: d.per_doc_evidence, reverse=True)

    # Trust 게이트(충돌/만료/민감차단) 우선 → evidence_score → LLM 자기판정 순. (포맷 무관 — 공통)
    gate = state.get("gate_flags")
    trust = state.get("trust_score")
    status = decide_answerability(
        documents,
        evidence_score=(trust.overall if trust else None),
        gate=gate,
        llm_status=llm_status,
    )

    # 인용 guard: ANSWERABLE인데 인용 출처 0건 → PARTIALLY로 강등
    if status == AnswerabilityStatus.ANSWERABLE and not sources:
        status, reason = AnswerabilityStatus.PARTIALLY_ANSWERABLE, NO_SOURCE_REASON
    else:
        reason = reason_for(status)

    # 비교 질의에서 미회수 target 버전이 있으면 사유에 명시 (#108 — 회수율 coverage의 정직한 고지)
    missing = state.get("missing_versions", [])
    if missing and status == AnswerabilityStatus.PARTIALLY_ANSWERABLE:
        reason = f"{', '.join(missing)} 버전 문서를 찾지 못해 검색된 버전 기준으로만 답변합니다."

    # 보류 상태는 본문 비움. 단 CONFLICTING/OUTDATED는 "왜 보류됐는지" 근거 문서를 보여줘야 하므로 sources 유지(P1).
    if status in _HOLD_STATUSES:
        held_sources = [] if status == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE else sources
        return _result(
            answer_format=answer_format,
            answer=FiveElements(),
            answer_text="",
            sources=held_sources,
            status=status,
            reason=reason,
        )
    return _result(
        answer_format=answer_format,
        answer=five,
        answer_text=answer_text,
        sources=sources,
        status=status,
        reason=reason,
    )
