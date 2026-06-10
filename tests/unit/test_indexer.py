import pytest

from app.config import Settings
from app.rag.chunker import ChildChunk
from app.rag.indexer import _point_id, index_children


def _child(chunk_id: str, content: str = "내용") -> ChildChunk:
    return ChildChunk(
        chunk_id=chunk_id,
        parent_id="p",
        page_id="pg",
        page_title="제목",
        content=content,
        embedding_text="emb " + content,
        heading_path=["h"],
        chunk_index=0,
        token_count=1,
        overlap_from_previous=0,
        source_url="http://x",
        space_key="OnRamp",
        last_modified="2026-06-01T00:00:00Z",
        hash="h",
        domain="장애대응",
    )


class _FakeEmbedder:
    dim = 3

    async def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class _FakeClient:
    def __init__(self):
        self.upserted = None
        self.upsert_calls = 0

    def get_collections(self):
        return type("R", (), {"collections": []})()

    def create_collection(self, **kwargs):
        pass

    def create_payload_index(self, *args, **kwargs):
        pass

    def upsert(self, collection_name, points):
        self.upserted = points if self.upserted is None else self.upserted + points
        self.upsert_calls += 1


def test_point_id_stable_and_unique():
    assert _point_id("pg_000") == _point_id("pg_000")
    assert _point_id("pg_000") != _point_id("pg_001")


@pytest.mark.asyncio
async def test_index_children_upserts_with_clean_payload():
    client = _FakeClient()
    n = await index_children(
        [_child("pg_000"), _child("pg_001")],
        embedder=_FakeEmbedder(),
        client=client,
        settings=Settings(embedding_dim=3),
    )
    assert n == 2
    assert len(client.upserted) == 2
    payload = client.upserted[0].payload
    assert payload["content"] == "내용"
    assert payload["domain"] == "장애대응"
    assert "content_vector" not in payload
    assert "embedding_text" not in payload
    assert len(client.upserted[0].vector) == 3


@pytest.mark.asyncio
async def test_index_children_empty_returns_zero():
    assert await index_children([], embedder=_FakeEmbedder(), client=_FakeClient(), settings=Settings()) == 0


@pytest.mark.asyncio
async def test_index_children_batches_upserts(monkeypatch):
    """Qdrant payload 한도(32MB) 초과 방지 — 배치 크기 단위로 나눠 upsert 한다."""
    monkeypatch.setattr("app.rag.indexer.UPSERT_BATCH_SIZE", 2)
    client = _FakeClient()
    children = [_child(f"pg_{i:03d}") for i in range(5)]
    n = await index_children(children, embedder=_FakeEmbedder(), client=client, settings=Settings(embedding_dim=3))
    assert n == 5
    assert client.upsert_calls == 3  # 2+2+1
    assert len(client.upserted) == 5
