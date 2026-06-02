"""Answerability Status 판정 (P0/P1 공용 판단 경계).

P0: documents 결정론 floor + Answer LLM 자기판정(llm_status) + 인용 guard(node에서).
P1: Trust의 Final Evidence Score(evidence_score)·게이트(gate)가 들어오면 임계값/게이트로 분기.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.agents.state import AnswerabilityStatus, SourceDocument

ANSWERABLE_THRESHOLD = 0.80  # Final Evidence Score 기준
PARTIAL_THRESHOLD = 0.60

NO_SOURCE_REASON = "답변을 뒷받침할 인용 출처를 찾지 못했습니다."

_REASONS = {
    AnswerabilityStatus.PARTIALLY_ANSWERABLE: "관련 문서가 있으나 근거가 일부 부족합니다.",
    AnswerabilityStatus.NOT_ENOUGH_EVIDENCE: "관련 근거를 찾지 못해 답변을 보류합니다.",
    AnswerabilityStatus.CONFLICTING_EVIDENCE: "문서 간 내용이 충돌해 확인이 필요합니다.",
    AnswerabilityStatus.OUTDATED_EVIDENCE: "최신 문서를 찾지 못해 제한적으로만 답변합니다.",
}


@dataclass
class GateFlags:
    """P1 게이트 신호 (Trust가 채움). P0에선 미사용."""

    conflicting: bool = False  # 동등 권위 문서 간 내용 충돌
    deprecated_only: bool = False  # deprecated/archived 문서만 검색됨
    sensitive_block: bool = False  # 고위험 민감정보 차단


def reason_for(status: AnswerabilityStatus) -> str:
    """상태별 기본 안내 메시지 (ANSWERABLE은 빈 문자열)."""
    return _REASONS.get(status, "")


def decide_answerability(
    documents: list[SourceDocument],
    *,
    evidence_score: float | None = None,
    gate: GateFlags | None = None,
    llm_status: AnswerabilityStatus | None = None,
) -> AnswerabilityStatus:
    """최종 Answerability Status를 결정한다 (P0/P1 공용).

    우선순위: 게이트(P1) → 무근거 floor → 점수(P1) → LLM 자기판정(P0).
    인용 source 0건에 대한 ANSWERABLE 강등은 sources를 아는 node에서 처리한다.
    """
    # 1) 게이트 (P1) — Trust가 채울 때만
    if gate is not None:
        if gate.conflicting:
            return AnswerabilityStatus.CONFLICTING_EVIDENCE
        if gate.deprecated_only:
            return AnswerabilityStatus.OUTDATED_EVIDENCE
        if gate.sensitive_block:
            return AnswerabilityStatus.NOT_ENOUGH_EVIDENCE

    # 2) 결정론 floor — 문서 없으면 근거 부족
    if not documents:
        return AnswerabilityStatus.NOT_ENOUGH_EVIDENCE

    # 3) 점수 기반 (P1) — Final Evidence Score
    if evidence_score is not None:
        if evidence_score >= ANSWERABLE_THRESHOLD:
            return AnswerabilityStatus.ANSWERABLE
        if evidence_score >= PARTIAL_THRESHOLD:
            return AnswerabilityStatus.PARTIALLY_ANSWERABLE
        return AnswerabilityStatus.NOT_ENOUGH_EVIDENCE

    # 4) P0 — LLM 자기판정 (없으면 보수적으로 보류)
    return llm_status or AnswerabilityStatus.NOT_ENOUGH_EVIDENCE
