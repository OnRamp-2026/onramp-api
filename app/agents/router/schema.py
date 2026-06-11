"""Router Agent 출력 스키마."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.agents.state import Domain, UseCase

# RouterOutput 계약 버전. domains 계약/검증 규칙이 바뀌면 올린다.
# 평가 예측 캐시의 stale 판정 키로 쓰여, 계약이 바뀌면 옛 캐시를 무효화한다.
SCHEMA_VERSION = "1"


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

    @model_validator(mode="before")
    @classmethod
    def _clear_unanswerable_domains(cls, data: Any) -> Any:
        # UNANSWERABLE은 검색 자체를 안 하므로 domains 내용과 무관 — **필드 검증(max_length·enum)
        # 전에** 미리 비운다. mode="after"에서 비우면 그 전에 필드 검증이 먼저 실패할 수 있고,
        # route_node가 ValidationError를 잡아 SEARCH fallback하므로 "답변불가 질문이 검색으로
        # 전환되는" 안전성 버그가 된다. 입력 dict는 직접 mutate하지 않고 복사해서 수정한다.
        # (use_case 자체가 잘못된 값이면 여기서 손대지 않고 평소대로 ValidationError가 나게 둔다.)
        if isinstance(data, dict) and data.get("use_case") in (
            UseCase.UNANSWERABLE,
            UseCase.UNANSWERABLE.value,
        ):
            data = dict(data)
            data["domains"] = []
        return data

    @model_validator(mode="after")
    def _check_domains(self) -> RouterOutput:
        # SEARCH만 도메인 검증 대상. UNANSWERABLE은 위 before validator에서 domains=[]로
        # 정규화되므로 여기 중복/빈 배열 검사를 적용하지 않는다(적용하면 빈 배열→ValidationError→
        # SEARCH fallback으로 다시 답변불가 질문이 검색으로 새게 된다).
        if self.use_case != UseCase.SEARCH:
            return self
        if len(self.domains) != len(set(self.domains)):
            raise ValueError(f"domains 중복 금지: {self.domains}")
        if not self.domains:
            # 정상 SEARCH 출력은 최소 1개 도메인 필요. 빈 배열이면 LLM 출력 결함 →
            # 파싱 실패로 처리해 route_node가 fallback(무가산 검색)하게 한다.
            # (저신뢰 도메인 비움은 route_node가 별도로 처리)
            raise ValueError("use_case=SEARCH면 domains는 최소 1개여야 합니다")
        return self
