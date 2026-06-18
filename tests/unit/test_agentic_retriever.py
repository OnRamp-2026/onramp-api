from qdrant_client.models import ScoredPoint

from app.agents.retriever import agentic as agentic_mod
from app.agents.retriever.agentic import AgenticRetrievalFallbackError, run_agentic_search
from app.agents.retriever.tools import SearchToolContext
from app.config import Settings
from app.services.llm_selector import ToolCall, ToolLLMResponse


def _hit(chunk_id: str, score: float) -> ScoredPoint:
    return ScoredPoint(
        id=chunk_id,
        version=0,
        score=score,
        payload={"chunk_id": chunk_id, "content": f"content-{chunk_id}", "page_title": chunk_id},
    )


async def test_agentic_search_fuses_rankings_from_different_iterations(monkeypatch) -> None:
    responses = iter(
        [
            ToolLLMResponse(content="", tool_calls=[ToolCall(id="1", name="search_dense", arguments={"query": "q1"})]),
            ToolLLMResponse(content="", tool_calls=[ToolCall(id="2", name="search_bm25", arguments={"query": "q2"})]),
        ]
    )

    async def fake_llm(*args, **kwargs):
        return next(responses)

    async def fake_execute(tool_name: str, query: str, *, context: SearchToolContext):
        if tool_name == "search_dense":
            return [_hit("shared", 0.9), _hit("dense-only", 0.8)]
        return [_hit("shared", 9.0), _hit("bm25-only", 8.0)]

    monkeypatch.setattr(agentic_mod, "call_llm_with_tools", fake_llm)
    monkeypatch.setattr(agentic_mod, "execute_search_tool", fake_execute)
    settings = Settings(retriever_agentic_max_iterations=2)
    context = SearchToolContext("tenant", None, None, 20, settings)

    result = await run_agentic_search("original", model="gpt-4o-mini", context=context)

    assert result.metadata["rrf_applied"] is True
    assert result.metadata["ranking_list_count"] == 2
    assert result.hits[0].id == "shared"
    assert [call["tool"] for call in result.metadata["calls"]] == ["search_dense", "search_bm25"]


async def test_agentic_search_dedupes_same_tool_query_and_filters(monkeypatch) -> None:
    response = ToolLLMResponse(
        content="",
        tool_calls=[
            ToolCall(id="1", name="search_dense", arguments={"query": "same query"}),
            ToolCall(id="2", name="search_dense", arguments={"query": " same   query "}),
        ],
    )
    executed: list[tuple[str, str]] = []

    async def fake_llm(*args, **kwargs):
        return response

    async def fake_execute(tool_name: str, query: str, *, context: SearchToolContext):
        executed.append((tool_name, query))
        return [_hit("c1", 0.8)]

    monkeypatch.setattr(agentic_mod, "call_llm_with_tools", fake_llm)
    monkeypatch.setattr(agentic_mod, "execute_search_tool", fake_execute)
    settings = Settings(retriever_agentic_max_iterations=1)
    context = SearchToolContext("tenant", None, None, 20, settings)

    result = await run_agentic_search("original", model="", context=context)

    assert executed == [("search_dense", "same query")]
    assert result.metadata["duplicate_calls"] == 1
    assert result.metadata["rrf_applied"] is False


async def test_agentic_search_without_valid_tool_calls_requests_fallback(monkeypatch) -> None:
    async def fake_llm(*args, **kwargs):
        return ToolLLMResponse(content="검색 없이 종료", tool_calls=[])

    monkeypatch.setattr(agentic_mod, "call_llm_with_tools", fake_llm)
    settings = Settings(retriever_agentic_max_iterations=1)
    context = SearchToolContext("tenant", None, None, 20, settings)

    try:
        await run_agentic_search("original", model="", context=context)
    except AgenticRetrievalFallbackError as exc:
        assert exc.reason == "no_search_results"
    else:
        raise AssertionError("fallback expected")
