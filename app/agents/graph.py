"""
LangGraph 워크플로우 그래프 정의.

Sprint 2 P0 경로:
    Router → Retriever → Answer → END
    Router(UNANSWERABLE) → END  (검색 생략)

Sprint 3 P1 추가 예정:
    Answer → Trust → (재검색 or END)
    Cache Manager (Redis L1)
"""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.answer.node import answer_node
from app.agents.retriever.node import retrieve_node
from app.agents.router.node import route_node
from app.agents.state import AgentState, UseCase

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

    # Sprint 2 P0 노드 등록
    graph.add_node("router", route_node)
    graph.add_node("retriever", retrieve_node)
    graph.add_node("answer", answer_node)

    # 엣지 연결: Router → (조건 분기) → Retriever → Answer → END
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "retriever": "retriever",
            "end": END,
        },
    )
    graph.add_edge("retriever", "answer")
    graph.add_edge("answer", END)

    # -----------------------------------------------------------------
    # Sprint 3 P1: Trust Agent(Evidence Confidence) + 재검색 루프
    #
    #   Trust는 retriever와 answer "사이"에 위치한다.
    #   검색된 문서를 5축으로 채점해 근거가 부족하면 retriever로 되돌려
    #   재검색하고(재시도 한도 내), 충분하면 answer로 진행한다.
    #   Answerability Status(최종 처리 방식) 판단은 answer가 담당한다.
    #
    #   ※ 이 배선을 켤 때 위의 Sprint 2 엣지(retriever → answer)는 제거한다.
    # -----------------------------------------------------------------
    # graph.add_node("trust", trust_node)
    # graph.add_edge("retriever", "trust")        # retriever → trust (answer 앞)
    # graph.add_conditional_edges(
    #     "trust",
    #     trust_decision,
    #     {
    #         "retriever": "retriever",  # 근거 부족 & 재시도 가능 → 재검색
    #         "answer": "answer",        # 근거 충분 → 답변 생성
    #     },
    # )
    # graph.add_edge("answer", END)
    # -----------------------------------------------------------------

    return graph.compile()


# 모듈 레벨에서 컴파일 — chat_service에서 import해서 사용
compiled_graph = build_graph()
