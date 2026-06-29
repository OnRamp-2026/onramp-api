import pytest

from app.config import Settings
from app.rag.chunker import ChildChunk
from app.rag.indexer import _opensearch_document, _point_id, index_children


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
        site="apache",
        product_version="2.2",
        doc_key="apache:content-negotiation",
        is_eol=True,
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


class _FakeOpenSearchClient:
    def __init__(self):
        self.documents = None

    async def upsert_chunks(self, documents):
        self.documents = documents


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
    # 버전 계보 메타 (#94) — asdict 흐름으로 payload에 포함
    assert payload["site"] == "apache"
    assert payload["product_version"] == "2.2"
    assert payload["doc_key"] == "apache:content-negotiation"
    assert payload["is_eol"] is True
    assert payload["tenant_id"] == "onramp"
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


@pytest.mark.asyncio
async def test_index_children_batches_exact_multiple(monkeypatch):
    """청크 수가 배치 크기의 정확한 배수일 때 빈 배치 호출이 없어야 한다."""
    monkeypatch.setattr("app.rag.indexer.UPSERT_BATCH_SIZE", 2)
    client = _FakeClient()
    children = [_child(f"pg_{i:03d}") for i in range(4)]
    n = await index_children(children, embedder=_FakeEmbedder(), client=client, settings=Settings(embedding_dim=3))
    assert n == 4
    assert client.upsert_calls == 2  # 2+2 — 빈 3회차 없음
    assert len(client.upserted) == 4


@pytest.mark.asyncio
async def test_index_children_upserts_opensearch_when_enabled():
    client = _FakeClient()
    os_client = _FakeOpenSearchClient()
    settings = Settings(embedding_dim=3, bm25_search_enabled=True, auth_default_tenant="tenant-a")

    n = await index_children(
        [_child("pg_000")],
        embedder=_FakeEmbedder(),
        client=client,
        opensearch_client=os_client,
        settings=settings,
    )

    assert n == 1
    assert client.upserted is not None
    assert len(client.upserted) == 1
    assert client.upserted[0].payload["tenant_id"] == "tenant-a"
    assert os_client.documents[0]["tenant_id"] == "tenant-a"
    assert os_client.documents[0]["embedding_text"].startswith("emb")


@pytest.mark.asyncio
async def test_index_children_uses_explicit_tenant_and_source():
    client = _FakeClient()
    os_client = _FakeOpenSearchClient()
    settings = Settings(embedding_dim=3, bm25_search_enabled=True)

    await index_children(
        [_child("gh:repo#1_000")],
        embedder=_FakeEmbedder(),
        client=client,
        opensearch_client=os_client,
        settings=settings,
        tenant_id="tenant-a",
        source="github",
    )

    assert client.upserted[0].payload["tenant_id"] == "tenant-a"
    assert client.upserted[0].payload["source"] == "github"
    assert os_client.documents[0]["tenant_id"] == "tenant-a"
    assert os_client.documents[0]["source"] == "github"


def test_opensearch_document_normalizes_optional_lists():
    child = _child("pg_000")
    document = _opensearch_document(child, Settings(auth_default_tenant="tenant-a"))

    assert document["tenant_id"] == "tenant-a"
    assert document["block_types"] == []
    assert document["keywords"] == []
    assert document["tags"] == []
    assert document["code_languages"] == []
