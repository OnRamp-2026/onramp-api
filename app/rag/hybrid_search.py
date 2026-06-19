"""Dense(Qdrant) + BM25(OpenSearch) hybrid retrieval."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

import httpx
from qdrant_client.models import ScoredPoint

from app.agents.retriever.search import SearchFilters
from app.config import Settings
from app.db.opensearch import OpenSearchClient, get_opensearch
from app.rag.rrf import RankedItem, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

DenseSearchFn = Callable[..., Awaitable[list[ScoredPoint]]]


async def hybrid_search(
    query_text: str,
    query_vector: list[float],
    *,
    top_k: int,
    tenant_id: str | None = None,
    domain: str | None,
    source: str | None = None,
    filters: SearchFilters | None,
    settings: Settings,
    dense_search_fn: DenseSearchFn,
    opensearch_client: OpenSearchClient | None = None,
) -> list[ScoredPoint]:
    final_top_k = max(top_k, settings.hybrid_final_top_k)
    dense_limit = max(settings.hybrid_dense_top_k, final_top_k)
    bm25_limit = max(settings.hybrid_bm25_top_k, final_top_k)
    dense_hits = await dense_search_fn(
        query_vector,
        dense_limit,
        domain=domain,
        tenant_id=tenant_id,
        source=source,
        filters=filters,
        settings=settings,
    )
    try:
        bm25_hits = await (opensearch_client or get_opensearch()).search(
            query_text,
            top_k=bm25_limit,
            tenant_id=tenant_id or settings.auth_default_tenant,
            domain=domain,
            source=source,
            version=filters.version if filters else None,
            pinned_doc_keys=filters.pinned_doc_keys if filters else (),
            excluded_doc_keys=filters.excluded_doc_keys if filters else (),
        )
    except httpx.HTTPError:
        logger.exception("OpenSearch BM25 search failed; falling back to dense-only retrieval")
        bm25_hits = []

    dense_items = [
        RankedItem(
            id=str((point.payload or {}).get("chunk_id") or point.id),
            score=float(point.score),
            payload=point.payload or {},
        )
        for point in dense_hits
    ]
    bm25_items = [
        RankedItem(id=str(hit.payload.get("chunk_id") or hit.id), score=hit.score, payload=hit.payload)
        for hit in bm25_hits
    ]
    fused = reciprocal_rank_fusion(
        (("dense", dense_items), ("bm25", bm25_items)),
        k=settings.hybrid_rrf_k,
        limit=final_top_k,
    )
    return [
        ScoredPoint(
            id=item.id, version=0, score=item.score, payload={**item.payload, "_hybrid_scores": item.source_scores}
        )
        for item in fused
    ]
