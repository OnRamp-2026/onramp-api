"""
LangGraph 워크플로우 그래프 정의.

경로:
    Router → Retriever → Trust → Answer → END
    Router(UNANSWERABLE) → END           (검색 생략)
    Trust(근거 부족) → Retriever          (재검색 루프, max_retries 한도 내)
"""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.answer.node import answer_node
from app.agents.retriever.node import retrieve_node
from app.agents.router.node import route_node
from app.agents.state import AgentState, UseCase
from app.agents.trust.node import trust_decision, trust_node

# ---------------------------------------------------------------------------
# 조건부 엣지 함수
# ---------------------------------------------------------------------------


def route_decision(state: AgentState) -> str:
    """Router 결과에 따라 다음 노드를 결정한다.

    - UNANSWERABLE → 즉시 종료 (Retriever, Answer 생략)
    - SEARCH → Retriever로 진행
    """
    if state.get("use_case") == UseCase.UNANSWERABLE:
        return "end"
    return "retriever"


# ---------------------------------------------------------------------------
# 그래프 빌드
# ---------------------------------------------------------------------------


def build_graph() -> CompiledStateGraph:
    """StateGraph를 조립하고 컴파일한다.

    Returns:
        CompiledGraph: invoke() / ainvoke()로 실행 가능한 컴파일된 그래프
    """
    graph = StateGraph(AgentState)

    # 노드 등록
    graph.add_node("router", route_node)
    graph.add_node("retriever", retrieve_node)
    graph.add_node("trust", trust_node)
    graph.add_node("answer", answer_node)

    # Router → (조건 분기) → Retriever
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "retriever": "retriever",
            "end": END,
        },
    )

    # Retriever → Trust → (근거 부족 시 재검색 루프 | 충분하면 Answer)
    #   Trust가 5축 채점 + 관련성(top rerank<τ) 기준으로 재검색 여부 결정.
    #   재시도는 max_retries 한도 내(무한루프 방지), 재시도 시 domain 필터 해제.
    graph.add_edge("retriever", "trust")
    graph.add_conditional_edges(
        "trust",
        trust_decision,
        {
            "retriever": "retriever",  # 근거 부족 & 재시도 가능 → 재검색
            "answer": "answer",  # 근거 충분 → 답변 생성
        },
    )
    graph.add_edge("answer", END)

    return graph.compile()


# 모듈 레벨에서 컴파일 — chat_service에서 import해서 사용
compiled_graph = build_graph()
