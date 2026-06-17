"""Answer Agent 출력 스키마."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.agents.state import AnswerabilityStatus

# P0 LLM 자기판정은 answerable/partially/not_enough만 유효 (conflicting/outdated는 P1 Trust 게이트가 결정).
_SELF_JUDGE_STATUSES = {
    AnswerabilityStatus.ANSWERABLE.value,
    AnswerabilityStatus.PARTIALLY_ANSWERABLE.value,
    AnswerabilityStatus.NOT_ENOUGH_EVIDENCE.value,
}


def _coerce_text(value: object) -> str:
    # LLM이 단계 목록을 배열로 주는 경우가 흔함 → 줄바꿈으로 합쳐 문자열로 수용
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _coerce_self_judge_status(value: object) -> object:
    # 오탈자·대소문자, 그리고 LLM이 임의로 낸 conflicting/outdated는 보수적으로 NOT_ENOUGH로 매핑해
    # status 한 글자 때문에 본문이 통째로 버려지지 않게 한다.
    if isinstance(value, AnswerabilityStatus):
        value = value.value
    if isinstance(value, str) and value.strip().lower() in _SELF_JUDGE_STATUSES:
        return value.strip().lower()
    return AnswerabilityStatus.NOT_ENOUGH_EVIDENCE


class AnswerOutput(BaseModel):
    """Answer LLM 응답(JSON) 파싱 결과 — 5요소 + LLM 자기판정.

    answerability_status는 LLM의 자기판정(P0 하이브리드 신호)일 뿐이며,
    최종 status는 node의 decide_answerability + 인용 guard가 결정한다.
    """

    situation: str = ""
    cause: str = ""
    evidence: str = ""
    solution: str = ""
    infra_context: str = ""
    answerability_status: AnswerabilityStatus = AnswerabilityStatus.ANSWERABLE
    answerability_reason: str = ""
    source_indices: list[int] = Field(default_factory=list)  # 근거로 사용한 문서 인덱스(0부터)

    @field_validator("situation", "cause", "evidence", "solution", "infra_context", mode="before")
    @classmethod
    def _coerce_str(cls, value: object) -> str:
        return _coerce_text(value)

    @field_validator("answerability_status", mode="before")
    @classmethod
    def _coerce_status(cls, value: object) -> object:
        return _coerce_self_judge_status(value)


class FreeformOutput(BaseModel):
    """Freeform 답변 LLM 응답(JSON) 파싱 결과 — 산문 본문 + LLM 자기판정.

    grounding·answerability·source_indices 계약은 AnswerOutput과 동일하며 본문만 answer_text 하나다(#191).
    """

    answer_text: str = ""
    answerability_status: AnswerabilityStatus = AnswerabilityStatus.ANSWERABLE
    answerability_reason: str = ""
    source_indices: list[int] = Field(default_factory=list)

    @field_validator("answer_text", mode="before")
    @classmethod
    def _coerce_str(cls, value: object) -> str:
        return _coerce_text(value)

    @field_validator("answerability_status", mode="before")
    @classmethod
    def _coerce_status(cls, value: object) -> object:
        return _coerce_self_judge_status(value)
