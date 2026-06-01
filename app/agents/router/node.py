"""
Router Agent 노드 (stub).

Sprint 2 구현 시 LLM을 호출해 5도메인 분류 + 쿼리 정제를 수행한다.
현재는 기본값을 반환하는 stub.

"""

from app.agents.state import AgentState, Domain, UseCase


def route_node(state: AgentState) -> dict:
    """사용자 질문을 분류하고 검색 쿼리를 정제한다.

    TODO: LLM 호출 → 5도메인 분류 + refined_query 생성
    """
    return {
        "use_case": UseCase.SEARCH,
        "domain": Domain.OPS_MANUAL,
        "refined_query": state["query"],
        "agent_trace": ["router"],
    }
