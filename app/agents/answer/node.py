"""
Answer Agent 노드 (stub).

Sprint 2 구현 시 LLM을 호출해 5요소 구조화 답변을 생성한다.
현재는 빈 FiveElements를 반환하는 stub.

"""

from app.agents.state import AgentState, FiveElements


def answer_node(_state: AgentState) -> dict:
    """검색된 문서를 바탕으로 5요소 구조화 답변을 생성한다.

    TODO: 문서 컨텍스트 조립 → LLM 호출 → 5요소 답변 + 답변불가 판정
    """
    return {
        "answer": FiveElements(),
        "sources": [],
        "is_answerable": True,
        "unanswerable_reason": "",
        "agent_trace": ["answer"],
    }
