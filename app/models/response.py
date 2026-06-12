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
    # 버전 계보 메타 (#108) — 비교 질의에서 두 출처가 어느 버전인지 제목 suffix 없이 구분 가능하게
    site: str = ""  # 문서 출처 사이트 (apache/kubernetes/... , 라벨 없는 문서는 "")
    product_version: str = ""  # 문서 버전 (v1.33/2.4/latest, 버전 무관 문서는 "")


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


class AssetResponse(BaseModel):
    """자산화 보고서 (draft/published)."""

    report_id: str
    title: str
    report: FiveElementsResponse
    category: str
    status: str  # "draft" | "published"
    confluence_url: str = ""
    created_at: str
    updated_at: str


class AssetApproveResponse(BaseModel):
    """POST /v1/asset/{id}/approve 응답."""

    report_id: str
    status: str  # "published"
    confluence_url: str
