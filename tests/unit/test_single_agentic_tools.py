import pytest

from app.agents.retriever import tools
from app.agents.retriever.tools import SearchToolContext, execute_tool
from app.config import Settings


def _context(**kwargs):
    return SearchToolContext(
        tenant_id=kwargs.get("tenant_id", "tenant-a"),
        domains=kwargs.get("domains", ("incident",)),
        candidate_doc_ids=frozenset(kwargs.get("candidate_doc_ids", {"doc-1"})),
        filters=None,
        top_k=5,
        settings=Settings(),
    )


@pytest.mark.asyncio
async def test_source_tool_passes_server_context(monkeypatch):
    captured = {}

    async def fake(query, *, source, context):
        captured.update(query=query, source=source, tenant=context.tenant_id)
        return []

    monkeypatch.setattr(tools, "_hybrid", fake)
    await execute_tool(
        "hybrid_search_by_source",
        {"query": "PR 10", "source": "github", "tenant_id": "attacker"},
        context=_context(),
    )
    assert captured == {"query": "PR 10", "source": "github", "tenant": "tenant-a"}


@pytest.mark.asyncio
async def test_source_tool_rejects_unknown_source():
    with pytest.raises(ValueError, match="source"):
        await execute_tool(
            "hybrid_search_by_source",
            {"query": "q", "source": "slack"},
            context=_context(),
        )


@pytest.mark.asyncio
async def test_document_tool_requires_prior_candidate():
    with pytest.raises(ValueError, match="doc_id"):
        await execute_tool(
            "opensearch_get_document",
            {"doc_id": "other"},
            context=_context(),
        )
