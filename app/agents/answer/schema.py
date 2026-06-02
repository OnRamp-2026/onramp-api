"""Answer Agent 출력 스키마."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.agents.state import AnswerabilityStatus


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
        # LLM이 단계 목록을 배열로 주는 경우가 흔함 → 줄바꿈으로 합쳐 문자열로 수용
        if isinstance(value, list):
            return "\n".join(str(item) for item in value)
        if value is None:
            return ""
        return str(value)
