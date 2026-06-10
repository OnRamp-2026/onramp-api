"""Qdrant dense 벡터 검색 (검색측)."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

import anyio
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, ScoredPoint

from app.config import Settings, get_settings
from app.db.qdrant import get_qdrant

FilterMode = Literal["hard", "hybrid", "soft"]


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


async def search_with_mode(
    query_vector: list[float],
    top_k: int,
    *,
    domain: str | None,
    mode: FilterMode,
    settings: Settings | None = None,
) -> list[ScoredPoint]:
    """도메인 필터 모드에 따라 검색 전략을 적용한다 (운영 retrieve_node·평가 어댑터 공용).

    hard:   도메인 필터만 (확장 없음)
    hybrid: filtered-first + 저품질(0건/최고 score 미달) 무필터 확장 + merge
    soft:   무필터 검색 (도메인 가산은 호출측 rerank에서)
    domain이 None이면 모드와 무관하게 무필터.
    """
    settings = settings or get_settings()
    if mode == "soft" or not domain:
        return await dense_search(query_vector, top_k, domain=None, settings=settings)

    hits = await dense_search(query_vector, top_k, domain=domain, settings=settings)
    if mode == "hard":
        return hits
    # hybrid — 저품질이면 무필터로 보완
    if _is_low_quality(hits, settings):
        extra = await dense_search(query_vector, top_k, domain=None, settings=settings)
        hits = _merge_hits(hits, extra)
    return hits


def _is_low_quality(hits: list[ScoredPoint], settings: Settings) -> bool:
    """결과가 없거나 최고 score가 retriever_domain_min_score 미만이면 저품질로 본다."""
    if not hits:
        return True
    return max(point.score for point in hits) < settings.retriever_domain_min_score


def _merge_hits(primary: list[ScoredPoint], extra: list[ScoredPoint]) -> list[ScoredPoint]:
    """두 검색 결과를 point.id로 합치고 중복은 높은 score를 남긴다."""
    by_id: dict[int | str | UUID, ScoredPoint] = {point.id: point for point in primary}
    for point in extra:
        existing = by_id.get(point.id)
        if existing is None or point.score > existing.score:
            by_id[point.id] = point
    return sorted(by_id.values(), key=lambda point: point.score, reverse=True)


# P1: hybrid_search(BM25/sparse + RRF) — 동일 시그니처로 추가 예정
