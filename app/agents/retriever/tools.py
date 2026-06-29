"""Single Agentic Retriever tools with server-enforced tenant/source boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client.models import ScoredPoint

from app.agents.retriever.search import SearchFilters, dense_search
from app.agents.state import Domain
from app.config import Settings
from app.db.opensearch import get_opensearch
from app.rag.embedder import get_embedder
from app.rag.hybrid_search import hybrid_search


@dataclass(frozen=True)
class SearchToolContext:
    tenant_id: str
    domains: tuple[str, ...]
    candidate_doc_ids: frozenset[str]
    filters: SearchFilters | None
    top_k: int
    settings: Settings


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "hybrid_search",
            "description": "Dense와 BM25 결과를 RRF로 합쳐 사내 문서 청크를 검색합니다.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hybrid_search_by_source",
            "description": "github 또는 confluence 출처로 제한해 hybrid 검색합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "source": {"type": "string", "enum": ["github", "confluence"]},
                },
                "required": ["query", "source"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "opensearch_get_document",
            "description": "앞선 incident 검색 결과에 포함된 문서의 원문을 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {"doc_id": {"type": "string"}},
                "required": ["doc_id"],
                "additionalProperties": False,
            },
        },
    },
]


async def _hybrid(query: str, *, source: str | None, context: SearchToolContext) -> list[ScoredPoint]:
    vector = await get_embedder().embed_query(query)
    domain = context.domains[0] if context.domains else None
    if context.settings.retriever_domain_filter_mode == "soft":
        domain = None
    return await hybrid_search(
        query,
        vector,
        top_k=context.top_k,
        tenant_id=context.tenant_id,
        domain=domain,
        source=source,
        filters=context.filters,
        settings=context.settings,
        dense_search_fn=dense_search,
    )


async def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    context: SearchToolContext,
) -> list[ScoredPoint]:
    if not context.tenant_id:
        raise ValueError("tenant_id가 필요합니다")
    if tool_name in {"hybrid_search", "hybrid_search_by_source"}:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ValueError("검색 query가 필요합니다")
        source = None
        if tool_name == "hybrid_search_by_source":
            source = str(arguments.get("source") or "")
            if source not in {"github", "confluence"}:
                raise ValueError("source는 github 또는 confluence여야 합니다")
        return await _hybrid(query, source=source, context=context)
    if tool_name == "opensearch_get_document":
        if Domain.INCIDENT.value not in context.domains:
            raise ValueError("원문 조회는 incident 질의에서만 허용됩니다")
        doc_id = str(arguments.get("doc_id") or "").strip()
        if not doc_id or doc_id not in context.candidate_doc_ids:
            raise ValueError("앞선 검색 결과의 doc_id만 조회할 수 있습니다")
        document = await get_opensearch().get_document(doc_id, tenant_id=context.tenant_id)
        if not document:
            return []
        payload = {
            "chunk_id": f"document:{doc_id}",
            "page_id": doc_id,
            "page_title": document.get("title", ""),
            "content": str(document.get("content", ""))[: context.settings.single_agentic_document_max_chars],
            "source_url": document.get("source_url", ""),
            "space_key": document.get("space_key", ""),
            "domain": document.get("domain", ""),
            "source": document.get("source", ""),
            "last_modified": document.get("last_modified", ""),
        }
        return [ScoredPoint(id=payload["chunk_id"], version=0, score=1.0, payload=payload)]
    raise ValueError(f"지원하지 않는 tool: {tool_name}")
