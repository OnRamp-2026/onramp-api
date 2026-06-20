"""
LangGraph 워크플로우 테스트.

검증 항목:
    1. 검색 질문 → Router → Retriever → Trust → Answer 순차 실행
    2. 답변불가 질문 → Router → 즉시 종료 (Retriever, Answer 생략)
    3. Trust 노드 배선 + 재검색 루프(max_retries 한도) 종료
    4. route_decision 분기 로직 단위 검증

retrieve_node가 async(임베딩/검색 I/O)이므로 그래프는 ainvoke로 실행한다.
검색 경로 테스트는 retriever 의존성(embedder/dense_search)을 mock해 네트워크 없이 검증한다.
"""

import json

import pytest

from app.agents.graph import build_graph, compiled_graph, retriever_to_next, route_decision, trust_to_next
from app.agents.state import AnswerabilityStatus, Domain, RetrievalPhase, UseCase
from app.config import get_settings


class _FakeEmbedder:
    async def embed_query(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """router LLM·retriever 임베더/검색·trust 재작성 LLM을 stub해 네트워크 없이 결정론적으로 동작시킨다."""

    async def _fake_router_llm(system_prompt, user_prompt, **kwargs):
        # 검색 경로로 분류, 질문을 그대로 refined_query로 echo
        return json.dumps({"use_case": "검색", "domains": ["manual"], "refined_query": user_prompt, "confidence": 0.9})

    async def _empty_search(*args, **kwargs):
        return []

    async def _fake_rewrite_llm(system_prompt, user_prompt, **kwargs):
        # trust 재검색 사다리(REWRITE_QUERY)의 쿼리 재작성 — llm_selector 경유 호출을 차단
        return f"재작성: {user_prompt}"

    monkeypatch.setattr("app.agents.router.node.call_llm", _fake_router_llm)
    monkeypatch.setattr("app.agents.retriever.node.get_embedder", lambda *a, **k: _FakeEmbedder())
    monkeypatch.setattr("app.agents.retriever.search.dense_search", _empty_search)
    monkeypatch.setattr("app.services.llm_selector.call_llm", _fake_rewrite_llm)


class TestGraphSearchFlow:
    """검색(SEARCH) 경로 테스트."""

    async def test_runs_router_retriever_trust_answer(self) -> None:
        """검색 질문이 Router→Retriever→Trust→Answer 경로를 통과하는지 확인한다.

        dense_search stub이 0건을 반환 → Trust가 매번 재검색(retriever→trust)을 유발하고,
        max_retries 한도에서 종료 후 answer로 수렴한다. (기대 trace는 config max_retries로 유도)
        """
        max_retries = get_settings().trust_max_retries
        result = await compiled_graph.ainvoke({"query": "EKS Pod 장애 해결법"})

        # router → (retriever → trust) × (max_retries+1) → answer
        expected = ["router", *(["retriever", "trust"] * (max_retries + 1)), "answer"]
        assert result["agent_trace"] == expected

    async def test_returns_expected_state_keys(self) -> None:
        """결과에 필수 State 키가 모두 존재하는지 확인한다."""
        result = await compiled_graph.ainvoke({"query": "EKS Pod 장애 해결법"})

        assert result["query"] == "EKS Pod 장애 해결법"
        assert result["use_case"] == UseCase.SEARCH
        # 0건 → 사다리 1행(REWRITE_QUERY, #108): 도메인은 보존하고 쿼리만 재작성한다
        # (구형 "도메인 해제"는 EXPAND_TOPICS 행 전용으로 이동)
        assert result["domains"] == [Domain.MANUAL]
        assert result["domain"] == Domain.MANUAL
        assert result["refined_query"] == "재작성: EKS Pod 장애 해결법"

    async def test_router_multidomain_reaches_retriever(self, monkeypatch) -> None:
        """Router 멀티도메인이 Retriever까지 전달되는지 검증 (재검색 없으면 domains 보존)."""

        async def _multi(system_prompt, user_prompt, **kwargs):
            return json.dumps(
                {"use_case": "검색", "domains": ["incident", "manual"], "refined_query": user_prompt, "confidence": 0.9}
            )

        monkeypatch.setattr("app.agents.router.node.call_llm", _multi)
        # max_retries=0 → Trust 재검색 없음 → 라우터 domains가 초기화되지 않고 보존
        result = await compiled_graph.ainvoke({"query": "장애 원인과 복구 절차", "max_retries": 0})

        assert result["domains"] == [Domain.INCIDENT, Domain.MANUAL]
        assert result["domain"] == Domain.INCIDENT  # 하위호환 = domains[0]

    async def test_empty_docs_holds_answer(self) -> None:
        """검색 결과 0건이면 Answer가 보류(NOT_ENOUGH)로 수렴하는지 확인한다."""
        result = await compiled_graph.ainvoke({"query": "테스트 질문"})

        # retriever mock이 빈 결과 → answer 결정론 floor → 보류
        assert result["answerability_status"] == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE
        assert result["answerability_reason"]  # 보류 사유 메시지
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
                "domains": [],  # UNANSWERABLE → 도메인 없음
                "domain": None,
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


class TestSingleAgenticDecision:
    def test_retriever_routes_complete_to_answer(self) -> None:
        assert (
            retriever_to_next({"retriever_strategy": "single_agentic", "retrieval_phase": RetrievalPhase.COMPLETE})
            == "answer"
        )

    def test_retriever_routes_searched_to_trust(self) -> None:
        assert (
            retriever_to_next({"retriever_strategy": "single_agentic", "retrieval_phase": RetrievalPhase.SEARCHED})
            == "trust"
        )

    def test_trust_routes_exhausted_to_answer(self) -> None:
        assert trust_to_next({"retriever_strategy": "single_agentic", "retry_count": 1, "max_retries": 1}) == "answer"


class TestGraphStructure:
    """그래프 구조 검증."""

    def test_has_trust_node(self) -> None:
        """Trust 노드가 그래프에 배선되어 있는지 확인한다."""
        graph = compiled_graph.get_graph()
        assert "trust" in graph.nodes

    def test_has_required_nodes(self) -> None:
        """필수 노드(router, retriever, trust, answer)가 존재하는지 확인한다."""
        graph = compiled_graph.get_graph()
        node_names = set(graph.nodes.keys())

        assert "router" in node_names
        assert "retriever" in node_names
        assert "trust" in node_names
        assert "answer" in node_names

    async def test_re_retrieve_loop_terminates(self) -> None:
        """근거가 계속 부족해도 max_retries 한도에서 루프가 종료되는지 확인한다.

        dense_search stub이 항상 0건 → 무한 재검색을 막고 answer로 수렴해야 한다.
        """
        max_retries = get_settings().trust_max_retries
        result = await compiled_graph.ainvoke({"query": "근거 없는 질문"})

        # trust가 정확히 max_retries+1회만 실행되고 종료 (config로 유도)
        assert result["agent_trace"].count("trust") == max_retries + 1
        assert result["agent_trace"][-1] == "answer"
        assert result["retry_count"] == max_retries
