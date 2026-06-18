"""Agentic Retriever 검색 도구.

LLM은 tool 이름과 query만 선택한다. tenant와 Trust 필터는 SearchToolContext가 강제한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from qdrant_client.models import ScoredPoint

from app.agents.retriever.search import SearchFilters, dense_search
from app.config import Settings
from app.db.opensearch import get_opensearch
from app.rag.embedder import get_embedder


@dataclass(frozen=True)
class SearchToolContext:
    tenant_id: str
    domain: str | None
    filters: SearchFilters | None
    top_k: int
    settings: Settings


async def search_dense(query: str, *, context: SearchToolContext) -> list[ScoredPoint]:
    """순수 Qdrant dense 검색. 기존 hybrid gate를 의도적으로 우회한다."""
    query_vector = await get_embedder().embed_query(query)
    return cast(
        list[ScoredPoint],
        await dense_search(
            query_vector,
            context.top_k,
            domain=context.domain,
            filters=context.filters,
            settings=context.settings,
        ),
    )


async def search_bm25(query: str, *, context: SearchToolContext) -> list[ScoredPoint]:
    """OpenSearch BM25 검색. tenant와 정밀 필터는 서버 context에서만 가져온다."""
    filters = context.filters
    hits = await get_opensearch().search(
        query,
        top_k=context.top_k,
        tenant_id=context.tenant_id,
        domain=context.domain,
        version=filters.version if filters else None,
        pinned_doc_keys=filters.pinned_doc_keys if filters else (),
        excluded_doc_keys=filters.excluded_doc_keys if filters else (),
    )
    return [
        ScoredPoint(
            id=hit.id,
            version=0,
            score=hit.score,
            payload=hit.payload,
        )
        for hit in hits
    ]


async def execute_search_tool(
    tool_name: str,
    query: str,
    *,
    context: SearchToolContext,
) -> list[ScoredPoint]:
    if tool_name == "search_dense":
        return await search_dense(query, context=context)
    if tool_name == "search_bm25":
        return await search_bm25(query, context=context)
    raise ValueError(f"지원하지 않는 검색 도구: {tool_name}")
