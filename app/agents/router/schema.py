"""Router Agent 출력 스키마."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from app.agents.state import Domain, UseCase


class RouterOutput(BaseModel):
    """Router LLM 응답(JSON) 파싱 전용 스키마.

    use_case는 SEARCH 또는 UNANSWERABLE만 — 자산화는 /v1/asset API로 분리되어
    Router가 판별하지 않는다.

    domains: 질의가 요구하는 도메인을 **순서 있는 리스트**로(최대 2). domains[0]=대표,
    domains[1]=추가 검색 의도. 우선순위는 배열 순서로만 표현한다(별도 필드 금지).

    주의: 이 모델은 **LLM이 반환하는 JSON**만 표현한다. route_node는 이 값을
    AgentState 부분집합으로 매핑하면서 LLM 출력이 아닌 필드(``agent_trace``,
    UNANSWERABLE 시 ``answerability_reason``)와 하위호환 파생값(``domain=domains[0]``)을 주입한다.
    """

    use_case: UseCase
    domains: list[Domain] = Field(default_factory=list, max_length=2)
    refined_query: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_domains(self) -> RouterOutput:
        if len(self.domains) != len(set(self.domains)):
            raise ValueError(f"domains 중복 금지: {self.domains}")
        if self.use_case == UseCase.UNANSWERABLE:
            # UNANSWERABLE이면 도메인 없음 (검색 자체를 하지 않음)
            self.domains = []
        elif not self.domains:
            # 정상 SEARCH 출력은 최소 1개 도메인 필요. 빈 배열이면 LLM 출력 결함 →
            # 파싱 실패로 처리해 route_node가 fallback(무가산 검색)하게 한다.
            # (저신뢰 도메인 비움은 route_node가 별도로 처리)
            raise ValueError("use_case=SEARCH면 domains는 최소 1개여야 합니다")
        return self
