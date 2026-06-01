"""
LangGraph 워크플로우 테스트.

검증 항목:
    1. 검색 질문 → Router → Retriever → Answer 순차 실행
    2. 답변불가 질문 → Router → 즉시 종료 (Retriever, Answer 생략)
    3. Sprint 2 그래프에 Trust 노드 미포함
    4. route_decision 분기 로직 단위 검증

retrieve_node가 async(임베딩/검색 I/O)이므로 그래프는 ainvoke로 실행한다.
검색 경로 테스트는 retriever 의존성(embedder/dense_search)을 mock해 네트워크 없이 검증한다.
"""

import json

import pytest

from app.agents.graph import build_graph, compiled_graph, route_decision
from app.agents.state import AnswerabilityStatus, Domain, UseCase


class _FakeEmbedder:
    async def embed_query(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """router LLM·retriever 임베더/검색을 stub해 네트워크 없이 결정론적으로 동작시킨다."""

    async def _fake_router_llm(system_prompt, user_prompt, **kwargs):
        # 검색 경로로 분류, 질문을 그대로 refined_query로 echo
        return json.dumps({"use_case": "검색", "domain": "운영매뉴얼", "refined_query": user_prompt, "confidence": 0.9})

    async def _empty_search(*args, **kwargs):
        return []

    monkeypatch.setattr("app.agents.router.node.call_llm", _fake_router_llm)
    monkeypatch.setattr("app.agents.retriever.node.get_embedder", lambda *a, **k: _FakeEmbedder())
    monkeypatch.setattr("app.agents.retriever.node.dense_search", _empty_search)


class TestGraphSearchFlow:
    """검색(SEARCH) 경로 테스트."""

    async def test_runs_router_retriever_answer(self) -> None:
        """검색 질문이 3개 Agent를 순서대로 통과하는지 확인한다."""
        result = await compiled_graph.ainvoke({"query": "EKS Pod 장애 해결법"})

        assert result["agent_trace"] == ["router", "retriever", "answer"]

    async def test_returns_expected_state_keys(self) -> None:
        """결과에 필수 State 키가 모두 존재하는지 확인한다."""
        result = await compiled_graph.ainvoke({"query": "EKS Pod 장애 해결법"})

        assert result["query"] == "EKS Pod 장애 해결법"
        assert result["use_case"] == UseCase.SEARCH
        assert result["domain"] == Domain.OPS_MANUAL
        assert result["refined_query"] == "EKS Pod 장애 해결법"

    async def test_stub_returns_default_answer(self) -> None:
        """stub이 기본 답변 구조를 반환하는지 확인한다."""
        result = await compiled_graph.ainvoke({"query": "테스트 질문"})

        assert result["answerability_status"] == AnswerabilityStatus.ANSWERABLE
        assert result["answerability_reason"] == ""
        assert result["documents"] == []
        assert result["sources"] == []


class TestGraphUnanswerableFlow:
    """답변불가(UNANSWERABLE) 경로 테스트."""

    async def test_skips_retriever_and_answer(self, monkeypatch) -> None:
        """답변불가 판정 시 Retriever와 Answer를 건너뛰는지 확인한다."""

        def unanswerable_route(state: dict) -> dict:
            """검색 범위 밖 질문을 retrieve 전에 차단하는 route_node 대체 스텁."""
            return {
                "use_case": UseCase.UNANSWERABLE,
                "domain": Domain.OPS_MANUAL,
                "refined_query": state["query"],
                "answerability_reason": "사내 지식 범위 밖 질문입니다.",
                "agent_trace": ["router"],
            }

        monkeypatch.setattr("app.agents.graph.route_node", unanswerable_route)

        graph = build_graph()
        result = await graph.ainvoke({"query": "오늘 점심 뭐 먹지"})

        # Router만 실행, Retriever·Answer 미실행
        assert result["agent_trace"] == ["router"]
        assert result["use_case"] == UseCase.UNANSWERABLE
        assert result["answerability_reason"] == "사내 지식 범위 밖 질문입니다."
        assert "documents" not in result


class TestRouteDecision:
    """route_decision 분기 함수 단위 테스트."""

    def test_unanswerable_returns_end(self) -> None:
        """답변불가 use_case는 'end'로 분기한다."""
        assert route_decision({"use_case": UseCase.UNANSWERABLE}) == "end"

    def test_search_returns_retriever(self) -> None:
        """검색 use_case는 'retriever'로 분기한다."""
        assert route_decision({"use_case": UseCase.SEARCH}) == "retriever"

    def test_missing_use_case_defaults_to_retriever(self) -> None:
        """use_case가 없으면 검색으로 간주한다 (fallback)."""
        assert route_decision({}) == "retriever"


class TestGraphStructure:
    """Sprint 2 그래프 구조 검증."""

    def test_no_trust_node(self) -> None:
        """Sprint 2 그래프에 Trust 노드가 포함되지 않음을 확인한다."""
        graph = compiled_graph.get_graph()
        assert "trust" not in graph.nodes

    def test_has_required_nodes(self) -> None:
        """Sprint 2 필수 노드(router, retriever, answer)가 존재하는지 확인한다."""
        graph = compiled_graph.get_graph()
        node_names = set(graph.nodes.keys())

        assert "router" in node_names
        assert "retriever" in node_names
        assert "answer" in node_names
