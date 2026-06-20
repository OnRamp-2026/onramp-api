import pytest
from qdrant_client.models import ScoredPoint

from app.agents.retriever import agentic
from app.agents.retriever.agentic import merge_candidates, run_agentic_step
from app.agents.retriever.node import retrieve_node
from app.agents.state import RetrievalCandidate, RetrievalPhase
from app.config import Settings
from app.services.llm_selector import ToolCall, ToolResponse


def _candidate(score: float, chunk_id: str = "c1") -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        payload={"chunk_id": chunk_id, "content": "x", "page_id": "p1"},
        search_score=score,
        tool_name="hybrid_search",
        query="q",
    )


def test_merge_candidates_keeps_higher_score():
    merged = merge_candidates([_candidate(0.2)], [_candidate(0.8)], limit=10)
    assert len(merged) == 1
    assert merged[0].search_score == 0.8


@pytest.mark.asyncio
async def test_first_step_without_tool_falls_back_to_hybrid(monkeypatch):
    async def fake_llm(*args, **kwargs):
        return ToolResponse(content="done", tool_calls=[])

    async def fake_tool(name, arguments, *, context):
        assert name == "hybrid_search"
        return []

    monkeypatch.setattr(agentic, "call_llm_with_tools", fake_llm)
    monkeypatch.setattr(agentic, "execute_tool", fake_tool)
    out = await run_agentic_step(
        {"query": "원문", "tenant_id": "tenant-a", "retriever_strategy": "single_agentic"},
        Settings(),
    )
    assert out["retrieval_phase"] == RetrievalPhase.SEARCHED
    assert out["tool_trace"][0].fallback == "missing_initial_tool"


@pytest.mark.asyncio
async def test_second_step_without_tool_completes(monkeypatch):
    async def fake_llm(*args, **kwargs):
        return ToolResponse(content="enough", tool_calls=[])

    monkeypatch.setattr(agentic, "call_llm_with_tools", fake_llm)
    out = await run_agentic_step(
        {
            "query": "원문",
            "tenant_id": "tenant-a",
            "retriever_strategy": "single_agentic",
            "retrieval_candidates": [_candidate(0.5)],
        },
        Settings(),
    )
    assert out["retrieval_phase"] == RetrievalPhase.COMPLETE


@pytest.mark.asyncio
async def test_duplicate_query_is_not_executed(monkeypatch):
    async def fake_llm(*args, **kwargs):
        return ToolResponse(
            content="",
            tool_calls=[ToolCall(id="1", name="hybrid_search", arguments={"query": "same query"})],
        )

    async def fail_tool(*args, **kwargs):
        raise AssertionError("duplicate tool must not execute")

    monkeypatch.setattr(agentic, "call_llm_with_tools", fake_llm)
    monkeypatch.setattr(agentic, "execute_tool", fail_tool)
    out = await run_agentic_step(
        {
            "query": "원문",
            "tenant_id": "tenant-a",
            "retriever_strategy": "single_agentic",
            "retrieval_candidates": [
                RetrievalCandidate(
                    chunk_id="c1",
                    payload={"chunk_id": "c1", "content": "x", "page_id": "p1"},
                    search_score=0.5,
                    tool_name="hybrid_search",
                    query="same query",
                )
            ],
            "previous_queries": ["same query"],
        },
        Settings(),
    )
    assert out["retrieval_phase"] == RetrievalPhase.COMPLETE


@pytest.mark.asyncio
async def test_same_query_can_switch_from_confluence_to_github(monkeypatch):
    async def fake_llm(*args, **kwargs):
        return ToolResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="1",
                    name="hybrid_search_by_source",
                    arguments={"query": "same query", "source": "github"},
                )
            ],
        )

    called = []

    async def fake_tool(name, arguments, *, context):
        called.append((name, arguments["query"], arguments["source"]))
        return []

    existing = RetrievalCandidate(
        chunk_id="c1",
        payload={"chunk_id": "c1", "content": "x", "page_id": "p1", "source": "confluence"},
        search_score=0.5,
        tool_name="hybrid_search_by_source",
        query="same query",
    )
    monkeypatch.setattr(agentic, "call_llm_with_tools", fake_llm)
    monkeypatch.setattr(agentic, "execute_tool", fake_tool)

    out = await run_agentic_step(
        {
            "query": "원문",
            "tenant_id": "tenant-a",
            "retriever_strategy": "single_agentic",
            "retrieval_candidates": [existing],
            "previous_queries": ["same query"],
        },
        Settings(),
    )

    assert called == [("hybrid_search_by_source", "same query", "github")]
    assert out["retrieval_phase"] == RetrievalPhase.SEARCHED


@pytest.mark.asyncio
async def test_same_query_can_expand_source_search_to_all_documents(monkeypatch):
    async def fake_llm(*args, **kwargs):
        return ToolResponse(
            content="",
            tool_calls=[ToolCall(id="1", name="hybrid_search", arguments={"query": "same query"})],
        )

    called = []

    async def fake_tool(name, arguments, *, context):
        called.append((name, arguments["query"]))
        return []

    existing = RetrievalCandidate(
        chunk_id="c1",
        payload={"chunk_id": "c1", "content": "x", "page_id": "p1", "source": "confluence"},
        search_score=0.5,
        tool_name="hybrid_search_by_source",
        query="same query",
    )
    monkeypatch.setattr(agentic, "call_llm_with_tools", fake_llm)
    monkeypatch.setattr(agentic, "execute_tool", fake_tool)

    out = await run_agentic_step(
        {
            "query": "원문",
            "tenant_id": "tenant-a",
            "retriever_strategy": "single_agentic",
            "retrieval_candidates": [existing],
            "previous_queries": ["same query"],
        },
        Settings(),
    )

    assert called == [("hybrid_search", "same query")]
    assert out["retrieval_phase"] == RetrievalPhase.SEARCHED


@pytest.mark.asyncio
async def test_initial_source_search_also_collects_global_candidates(monkeypatch):
    async def fake_llm(*args, **kwargs):
        return ToolResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="1",
                    name="hybrid_search_by_source",
                    arguments={"query": "ArgoCD credential bootstrap", "source": "confluence"},
                )
            ],
        )

    called = []

    async def fake_tool(name, arguments, *, context):
        called.append((name, arguments))
        if name == "hybrid_search_by_source":
            return [
                ScoredPoint(
                    id="confluence-1",
                    version=0,
                    score=0.9,
                    payload={
                        "chunk_id": "confluence-1",
                        "page_id": "page-1",
                        "content": "일반 Secret 문서",
                        "source": "confluence",
                    },
                )
            ]
        return [
            ScoredPoint(
                id="github-1",
                version=0,
                score=0.8,
                payload={
                    "chunk_id": "github-1",
                    "page_id": "gh:gitops#5",
                    "content": "ArgoCD credential bootstrap 구현",
                    "source": "github",
                },
            )
        ]

    monkeypatch.setattr(agentic, "call_llm_with_tools", fake_llm)
    monkeypatch.setattr(agentic, "execute_tool", fake_tool)

    out = await run_agentic_step(
        {
            "query": "ArgoCD credential bootstrap 방식은?",
            "tenant_id": "tenant-a",
            "retriever_strategy": "single_agentic",
        },
        Settings(),
    )

    assert [(name, arguments.get("source")) for name, arguments in called] == [
        ("hybrid_search_by_source", "confluence"),
        ("hybrid_search", None),
    ]
    assert {candidate.chunk_id for candidate in out["retrieval_candidates"]} == {
        "confluence-1",
        "github-1",
    }
    assert [trace.tool for trace in out["tool_trace"]] == [
        "hybrid_search_by_source",
        "hybrid_search",
    ]


@pytest.mark.asyncio
async def test_retrieve_node_runs_single_agentic_and_reranks(monkeypatch):
    async def fake_llm(*args, **kwargs):
        return ToolResponse(
            content="",
            tool_calls=[ToolCall(id="1", name="hybrid_search", arguments={"query": "검색어"})],
        )

    async def fake_tool(*args, **kwargs):
        return [
            ScoredPoint(
                id="c1",
                version=0,
                score=0.8,
                payload={
                    "chunk_id": "c1",
                    "page_id": "p1",
                    "page_title": "문서",
                    "content": "근거",
                    "source": "confluence",
                },
            )
        ]

    class _Reranker:
        def rerank(self, query, candidates):
            return [(0.9, payload) for _, payload in candidates]

    monkeypatch.setattr(agentic, "call_llm_with_tools", fake_llm)
    monkeypatch.setattr(agentic, "execute_tool", fake_tool)
    monkeypatch.setattr("app.agents.retriever.node.get_reranker", lambda: _Reranker())
    monkeypatch.setattr("app.agents.retriever.node.get_lineages", lambda keys, **kwargs: {})

    out = await retrieve_node(
        {
            "query": "원문",
            "tenant_id": "tenant-a",
            "retriever_strategy": "single_agentic",
            "domains": [],
            "target_versions": [],
        }
    )

    assert out["retrieval_phase"] == RetrievalPhase.SEARCHED
    assert out["documents"][0].chunk_id == "c1"
    assert out["documents"][0].source == "confluence"
