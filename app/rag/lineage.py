"""버전 계보(lineage) 조회 — Qdrant facet 파생 (#94).

계보 = "doc_key의 문서가 색인에 어떤 product_version들로 존재하는가".
설계 문서(docs/Baemin/01_trust_agent_redesign.md) 3장은 Postgres 테이블을 제안했으나,
계보는 색인의 파생 정보일 뿐이므로 Qdrant facet으로 직접 집계한다 — 색인↔테이블
정합성 비용이 없고, "색인에 실존하는 버전"만 보이므로 retry_version 사전 확인
("더 새 버전이 색인에 있을 때만 재검색")과 정합이 정확히 맞는다.

동기 함수다 — 비동기 호출측(retriever/trust 노드)은 anyio.to_thread.run_sync로 감싼다.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.config import Settings, get_settings
from app.db.qdrant import get_qdrant

logger = logging.getLogger(__name__)

_FACET_LIMIT = 32  # 계보당 버전 수는 한 자릿수 — 여유 상한

# doc_key → (만료 시각 monotonic, 버전 집합)
_cache: dict[str, tuple[float, frozenset[str]]] = {}
_cache_lock = threading.Lock()


def fetch_lineage(
    doc_key: str, *, client: QdrantClient | None = None, settings: Settings | None = None
) -> frozenset[str]:
    """단일 doc_key의 계보(버전 집합)를 Qdrant facet으로 집계한다 (캐시 미사용)."""
    if not doc_key:
        return frozenset()
    client = client or get_qdrant()
    settings = settings or get_settings()
    response = client.facet(
        collection_name=settings.qdrant_collection,
        key="product_version",
        facet_filter=Filter(must=[FieldCondition(key="doc_key", match=MatchValue(value=doc_key))]),
        limit=_FACET_LIMIT,
    )
    return frozenset(str(hit.value) for hit in response.hits if hit.value)


def get_lineages(
    doc_keys: Iterable[str], *, client: QdrantClient | None = None, settings: Settings | None = None
) -> dict[str, frozenset[str]]:
    """doc_key 배치의 계보 조회. TTL 캐시 적용 — 미스만 facet 호출.

    doc_key=""(계보 없는 문서)는 조회를 생략하고 빈 집합을 돌려준다.
    facet 조회 실패는 빈 계보로 폴백(미캐싱) — 계보는 보조 신호라 version_fit 중립(0.5)으로
    강등될 뿐, 조회 장애가 요청 전체를 실패시키면 안 된다.
    """
    settings = settings or get_settings()
    ttl = settings.lineage_cache_ttl_seconds
    now = time.monotonic()
    result: dict[str, frozenset[str]] = {}
    misses: list[str] = []

    with _cache_lock:
        for key in dict.fromkeys(doc_keys):  # 중복 제거 + 순서 보존
            if not key:
                result[key] = frozenset()
                continue
            cached = _cache.get(key)
            if cached and cached[0] > now:
                result[key] = cached[1]
            else:
                misses.append(key)

    for key in misses:
        try:
            versions = fetch_lineage(key, client=client, settings=settings)
        except Exception:
            logger.warning("계보 facet 조회 실패 — 빈 계보 폴백(미캐싱): %s", key, exc_info=True)
            result[key] = frozenset()
            continue
        result[key] = versions
        if ttl > 0:
            with _cache_lock:
                _cache[key] = (time.monotonic() + ttl, versions)
    return result


def clear_lineage_cache() -> None:
    """캐시 초기화 — 테스트·재색인 직후 사용."""
    with _cache_lock:
        _cache.clear()
