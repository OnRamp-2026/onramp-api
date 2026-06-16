"""Qdrant dense 벡터 검색 (검색측)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

import anyio
from qdrant_client import QdrantClient
from qdrant_client.models import Condition, FieldCondition, Filter, MatchAny, MatchValue, ScoredPoint

from app.config import Settings, get_settings
from app.db.qdrant import get_qdrant

FilterMode = Literal["hard", "hybrid", "soft"]


@dataclass(frozen=True)
class SearchFilters:
    """재검색 사다리의 정밀 필터 (#108, 설계 6장).

    domain soft 정책과 달리 **모드와 무관하게 항상 적용**된다 — 버전/제외 필터는
    사다리 전략의 본질이지 가산 대상이 아니다.
    """

    version: str | None = None  # product_version == 값 (RETRY_VERSION)
    pinned_doc_keys: tuple[str, ...] = ()  # doc_key ∈ 집합 (RETRY_VERSION — 대상 계보 고정)
    excluded_doc_keys: tuple[str, ...] = ()  # doc_key ∉ 집합 (EXPAND_TOPICS — 새 주제 발견)

    def is_empty(self) -> bool:
        return not (self.version or self.pinned_doc_keys or self.excluded_doc_keys)


def _build_filter(domain: str | None, filters: SearchFilters | None) -> Filter | None:
    """domain 조건 + 사다리 필터를 하나의 Qdrant Filter로 합성한다."""
    must: list[Condition] = []
    must_not: list[Condition] = []
    if domain:
        must.append(FieldCondition(key="domain", match=MatchValue(value=domain)))
    if filters:
        if filters.version:
            must.append(FieldCondition(key="product_version", match=MatchValue(value=filters.version)))
        if filters.pinned_doc_keys:
            must.append(FieldCondition(key="doc_key", match=MatchAny(any=list(filters.pinned_doc_keys))))
        if filters.excluded_doc_keys:
            must_not.append(FieldCondition(key="doc_key", match=MatchAny(any=list(filters.excluded_doc_keys))))
    if not must and not must_not:
        return None
    return Filter(must=must or None, must_not=must_not or None)


async def dense_search(
    query_vector: list[float],
    top_k: int,
    *,
    domain: str | None = None,
    filters: SearchFilters | None = None,
    client: QdrantClient | None = None,
    settings: Settings | None = None,
) -> list[ScoredPoint]:
    """쿼리 벡터로 dense kNN 검색. domain/사다리 필터 지정 시 payload 필터."""
    client = client or get_qdrant()
    settings = settings or get_settings()

    query_filter = _build_filter(domain, filters)

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
    query_text: str = "",
    filters: SearchFilters | None = None,
    settings: Settings | None = None,
) -> list[ScoredPoint]:
    """도메인 필터 모드에 따라 검색 전략을 적용한다 (운영 retrieve_node·평가 어댑터 공용).

    hard:   도메인 필터만 (확장 없음)
    hybrid: filtered-first + 저품질(0건/최고 score 미달) 무필터 확장 + merge
    soft:   무필터 검색 (도메인 가산은 호출측 rerank에서)
    domain이 None이면 모드와 무관하게 무필터.
    사다리 필터(filters)는 **모든 모드·확장 경로에 항상 적용**된다 (#108).
    """
    settings = settings or get_settings()
    if settings.hybrid_search_enabled and query_text.strip():
        from app.rag.hybrid_search import hybrid_search

        effective_domain = None if mode == "soft" or not domain else domain
        return await hybrid_search(
            query_text,
            query_vector,
            top_k=top_k,
            domain=effective_domain,
            filters=filters,
            settings=settings,
            dense_search_fn=dense_search,
        )

    if mode == "soft" or not domain:
        return await dense_search(query_vector, top_k, domain=None, filters=filters, settings=settings)

    hits = await dense_search(query_vector, top_k, domain=domain, filters=filters, settings=settings)
    if mode == "hard":
        return hits
    # hybrid — 저품질이면 무필터로 보완 (사다리 필터는 유지)
    if _is_low_quality(hits, settings):
        extra = await dense_search(query_vector, top_k, domain=None, filters=filters, settings=settings)
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
