"""Router Agent 출력 스키마."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.agents.state import Domain, UseCase


class RouterOutput(BaseModel):
    """Router LLM 응답(JSON) 파싱 전용 스키마.

    use_case는 SEARCH 또는 UNANSWERABLE만 — 자산화는 /v1/asset API로 분리되어
    Router가 판별하지 않는다.

    주의: 이 모델은 **LLM이 반환하는 JSON**만 표현한다. route_node는 이 값을
    AgentState 부분집합으로 매핑하면서 LLM 출력이 아닌 필드(``agent_trace``,
    UNANSWERABLE 시 ``answerability_reason``)를 추가로 주입한다. 따라서 그 필드들은
    여기(RouterOutput)가 아니라 AgentState(state.py)의 계약에 속한다.
    """

    use_case: UseCase
    domain: Domain
    refined_query: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
