"""Router Agent 출력 스키마."""

from __future__ import annotations

from pydantic import BaseModel

from app.agents.state import Domain, UseCase


class RouterOutput(BaseModel):
    """Router LLM 응답(JSON) 파싱 결과.

    use_case는 SEARCH 또는 UNANSWERABLE만 — 자산화는 /v1/asset API로 분리되어
    Router가 판별하지 않는다.
    """

    use_case: UseCase
    domain: Domain
    refined_query: str
    confidence: float = 0.0  # 0.0 ~ 1.0
