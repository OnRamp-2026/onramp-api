"""Qdrant dense 벡터 검색 (검색측)."""

from __future__ import annotations

import anyio
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, ScoredPoint

from app.config import Settings, get_settings
from app.db.qdrant import get_qdrant


async def dense_search(
    query_vector: list[float],
    top_k: int,
    *,
    domain: str | None = None,
    client: QdrantClient | None = None,
    settings: Settings | None = None,
) -> list[ScoredPoint]:
    """쿼리 벡터로 dense kNN 검색. domain 지정 시 payload 필터."""
    client = client or get_qdrant()
    settings = settings or get_settings()

    query_filter = None
    if domain:
        query_filter = Filter(must=[FieldCondition(key="domain", match=MatchValue(value=domain))])

    # QdrantClient는 동기 → 이벤트 루프 비차단 위해 스레드로
    resp = await anyio.to_thread.run_sync(
        lambda: client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
    )
    return resp.points


# P1: hybrid_search(BM25/sparse + RRF) — 동일 시그니처로 추가 예정
