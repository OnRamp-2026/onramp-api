"""API 요청 스키마."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /v1/chat 요청."""

    query: str = Field(..., min_length=1, max_length=2000, description="사용자 질문")
    model: str = Field(default="", description="LLM 모델 (빈값이면 config 기본값)")


class AssetRequest(BaseModel):
    """POST /v1/asset 요청 — 회의 녹취 → 5요소 보고서 초안."""

    transcript: str = Field(..., min_length=10, max_length=50000, description="회의 녹취 텍스트")
    category: str = Field(default="회의록", description="카테고리 (장애대응, 운영매뉴얼, API명세, 회의록, 기획서)")
    title: str = Field(default="", description="보고서 제목 (빈값이면 자동 생성)")
    model: str = Field(default="", description="LLM 모델")


class AssetUpdateRequest(BaseModel):
    """PATCH /v1/asset/{id} 요청 — HITL 부분 수정 (보낸 필드만 반영)."""

    title: str | None = None
    category: str | None = None
    situation: str | None = None
    cause: str | None = None
    evidence: str | None = None
    solution: str | None = None
    infra_context: str | None = None
