"""API 응답 스키마."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceDoc(BaseModel):
    """답변에 인용된 출처 문서."""

    title: str = ""
    url: str = ""
    space_key: str = ""
    content_snippet: str = ""
    score: float = 0.0


class FiveElementsResponse(BaseModel):
    """5요소 구조화 답변."""

    situation: str = ""
    cause: str = ""
    evidence: str = ""
    solution: str = ""
    infra_context: str = ""


class ChatResponse(BaseModel):
    """POST /v1/chat 응답."""

    answer: FiveElementsResponse
    sources: list[SourceDoc] = Field(default_factory=list)
    answerability_status: str  # AnswerabilityStatus.value
    answerability_reason: str = ""
    domain: str = ""  # Domain.value (classifier 영문 키)
    model_used: str = ""
