"""ChildChunk → Qdrant 색인 헬퍼. C2 검색 테스트용 최소 구현 (C3가 IngestService와 결합)."""

from __future__ import annotations

from dataclasses import asdict
from uuid import NAMESPACE_URL, uuid5

import anyio
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from app.config import Settings, get_settings
from app.db.qdrant import ensure_collection, get_qdrant
from app.rag.chunker import ChildChunk
from app.rag.embedder import Embedder, get_embedder


def _point_id(chunk_id: str) -> str:
    # chunk_id("{page_id}_{idx:03d}")는 Qdrant ID로 부적합 → UUID5 (멱등)
    return str(uuid5(NAMESPACE_URL, chunk_id))


def _payload(child: ChildChunk) -> dict:
    data = asdict(child)
    data.pop("content_vector", None)  # 벡터는 payload 아님
    data.pop("embedding_text", None)  # 임베딩 입력 — 검색/표시엔 불필요
    return data


async def index_children(
    children: list[ChildChunk],
    *,
    embedder: Embedder | None = None,
    client: QdrantClient | None = None,
    settings: Settings | None = None,
) -> int:
    """ChildChunk를 embedding_text로 임베딩해 Qdrant upsert. 반환: upsert 수."""
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
    await anyio.to_thread.run_sync(lambda: client.upsert(collection_name=settings.qdrant_collection, points=points))
    return len(points)
