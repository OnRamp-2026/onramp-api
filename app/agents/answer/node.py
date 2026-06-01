"""
Answer Agent 노드 (stub).

Sprint 2 구현 시 LLM을 호출해 5요소 구조화 답변을 생성한다.
현재는 빈 FiveElements를 반환하는 stub.

"""

from app.agents.state import AgentState, AnswerabilityStatus, FiveElements


def answer_node(state: AgentState) -> dict:
    """Evidence Confidence 점수를 바탕으로 답변 가능성을 판정하고 5요소 답변을 생성한다.

    TODO: trust_score(Final Evidence Score) → Answerability Status 판단 →
          문서 컨텍스트 조립 → LLM 호출 → 5요소 답변 생성 또는 보류
    """
    _ = state
    return {
        "answer": FiveElements(),
        "sources": [],
        "answerability_status": AnswerabilityStatus.ANSWERABLE,
        "answerability_reason": "",
        "agent_trace": ["answer"],
    }
