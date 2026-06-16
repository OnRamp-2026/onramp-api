"""ChildChunk → Qdrant 색인 헬퍼. C2 검색 테스트용 최소 구현 (C3가 IngestService와 결합)."""

from __future__ import annotations

import logging
from dataclasses import asdict
from functools import partial
from uuid import NAMESPACE_URL, uuid5

import anyio
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from app.config import Settings, get_settings
from app.db.opensearch import OpenSearchClient, get_opensearch
from app.db.qdrant import ensure_collection, get_qdrant
from app.rag.chunker import ChildChunk
from app.rag.embedder import Embedder, get_embedder

logger = logging.getLogger(__name__)

UPSERT_BATCH_SIZE = 256  # 청크당 payload+벡터 ~20KB → 배치당 ~5MB (Qdrant 한도 32MB 대비 여유)


def _point_id(chunk_id: str) -> str:
    # chunk_id("{page_id}_{idx:03d}")는 Qdrant ID로 부적합 → UUID5 (멱등)
    return str(uuid5(NAMESPACE_URL, chunk_id))


def _payload(child: ChildChunk) -> dict:
    data = asdict(child)
    data.pop("content_vector", None)  # 벡터는 payload 아님
    data.pop("embedding_text", None)  # 임베딩 입력 — 검색/표시엔 불필요
    return data


def _opensearch_document(child: ChildChunk, settings: Settings) -> dict:
    data = asdict(child)
    data.pop("content_vector", None)
    data["tenant_id"] = settings.auth_default_tenant
    data["block_types"] = child.block_types or []
    data["keywords"] = child.keywords or []
    data["tags"] = child.tags or []
    data["code_languages"] = child.code_languages or []
    return data


async def index_children(
    children: list[ChildChunk],
    *,
    embedder: Embedder | None = None,
    client: QdrantClient | None = None,
    opensearch_client: OpenSearchClient | None = None,
    settings: Settings | None = None,
) -> int:
    """ChildChunk를 embedding_text로 임베딩해 Qdrant/OpenSearch upsert. 반환: upsert 수."""
    if not children:
        return 0
    embedder = embedder or get_embedder()
    client = client or get_qdrant()
    settings = settings or get_settings()

    await anyio.to_thread.run_sync(lambda: ensure_collection(client, settings))
    vectors = await embedder.embed_documents([c.embedding_text for c in children])
    points = [
        PointStruct(id=_point_id(c.chunk_id), vector=vec, payload=_payload(c))
        for c, vec in zip(children, vectors, strict=True)
    ]
    # Qdrant JSON payload 한도(32MB) 초과 방지 — 전체 적재(수천 청크)는 단일 upsert가 불가
    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[start : start + UPSERT_BATCH_SIZE]
        await anyio.to_thread.run_sync(partial(client.upsert, collection_name=settings.qdrant_collection, points=batch))
    if settings.bm25_search_enabled:
        os_client = opensearch_client or get_opensearch()
        documents = [_opensearch_document(child, settings) for child in children]
        try:
            await os_client.upsert_chunks(documents)
        except Exception:
            logger.exception("OpenSearch upsert failed; proceeding with Qdrant-only indexing")
    return len(points)
