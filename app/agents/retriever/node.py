"""
Retriever Agent 노드 (stub).

Sprint 2 구현 시 Qdrant Dense Search + bge-reranker-v2-m3 리랭킹을 수행한다.
현재는 빈 문서 리스트를 반환하는 stub.

"""

from app.agents.state import AgentState


def retrieve_node(state: AgentState) -> dict:
    """정제된 쿼리로 문서를 검색하고 리랭킹한다.

    TODO: Qdrant Dense Search → Cross-Encoder Reranker → Top-N 반환
    """
    return {
        "documents": [],
        "agent_trace": ["retriever"],
    }
