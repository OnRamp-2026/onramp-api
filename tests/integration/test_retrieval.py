"""C2 검색 통합 테스트 — 실 Qdrant. index_children로 시드 후 dense_search 검증.

리랭커 실모델(torch)은 제외 — 색인·검색·도메인 필터만 검증. 미가동 시 skip.
"""

import hashlib

import pytest
from qdrant_client import QdrantClient

from app.agents.retriever.search import dense_search
from app.config import Settings
from app.rag.chunker import ChildChunk
from app.rag.indexer import index_children

COLLECTION = "onramp_c2_itest"
DIM = 8


def _vec(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [digest[i % len(digest)] / 255.0 for i in range(DIM)]


class _DeterministicEmbedder:
    dim = DIM

    async def embed_documents(self, texts):
        return [_vec(t) for t in texts]

    async def embed_query(self, text):
        return _vec(text)


def _child(idx: int, domain: str) -> ChildChunk:
    text = f"{domain} 문서 본문 {idx}"
    return ChildChunk(
        chunk_id=f"pg{idx}_000",
        parent_id=f"par{idx}",
        page_id=f"pg{idx}",
        page_title=f"{domain} 제목 {idx}",
        content=text,
        embedding_text=text,
        heading_path=[domain],
        chunk_index=0,
        token_count=5,
        overlap_from_previous=0,
        source_url=f"http://x/{idx}",
        space_key="OnRamp",
        last_modified="2026-06-01T00:00:00Z",
        hash=f"h{idx}",
        domain=domain,
    )


@pytest.fixture
def settings():
    return Settings(qdrant_collection=COLLECTION, embedding_dim=DIM)


@pytest.fixture
def qclient(settings):
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    try:
        client.get_collections()
    except Exception:
        pytest.skip("Qdrant 미가동 (make up 필요)")
    if COLLECTION in {c.name for c in client.get_collections().collections}:
        client.delete_collection(COLLECTION)
    yield client
    client.delete_collection(COLLECTION)
    client.close()


@pytest.mark.asyncio
async def test_index_and_domain_filter(qclient, settings):
    children = [
        _child(0, "장애대응"),
        _child(1, "장애대응"),
        _child(2, "API명세"),
        _child(3, "API명세"),
    ]
    n = await index_children(children, embedder=_DeterministicEmbedder(), client=qclient, settings=settings)
    assert n == 4

    qvec = _vec("장애대응 문서 본문 0")

    # 도메인 필터 → 해당 도메인만
    filtered = await dense_search(qvec, top_k=10, domain="장애대응", client=qclient, settings=settings)
    assert filtered
    assert all(p.payload["domain"] == "장애대응" for p in filtered)

    # 필터 없음 → 두 도메인 모두 후보
    unfiltered = await dense_search(qvec, top_k=10, client=qclient, settings=settings)
    assert {p.payload["domain"] for p in unfiltered} == {"장애대응", "API명세"}


@pytest.mark.asyncio
async def test_empty_domain_returns_nothing(qclient, settings):
    await index_children([_child(0, "장애대응")], embedder=_DeterministicEmbedder(), client=qclient, settings=settings)
    # 색인에 없는 도메인 필터 → 0건 (node 레벨에서 무필터 폴백)
    hits = await dense_search(_vec("q"), top_k=10, domain="회의록", client=qclient, settings=settings)
    assert hits == []
