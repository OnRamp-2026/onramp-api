"""API 요청 스키마."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /v1/chat 요청."""

    query: str = Field(..., min_length=1, max_length=2000, description="사용자 질문")
    model: str = Field(default="", description="LLM 모델 (빈값이면 config 기본값)")
