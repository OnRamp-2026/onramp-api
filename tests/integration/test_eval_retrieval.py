"""검색 평가 어댑터 통합 테스트 — 실 Qdrant 시드 후 ranked_chunk_ids 검증.

리랭커(torch) 제외 — dense 모드만. Qdrant 미가동 시 skip.
패턴: tests/integration/test_retrieval.py (_DeterministicEmbedder + index_children).
"""

import hashlib

import pytest
from qdrant_client import QdrantClient

from app.agents.retriever import search as search_mod
from app.config import Settings
from app.eval import retrieval_adapter as adapter
from app.rag.chunker import ChildChunk
from app.rag.indexer import index_children

COLLECTION = "onramp_eval_itest"
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
    except Exception as exc:  # 연결 실패만 skip, 그 외(회귀)는 그대로 실패
        msg = str(exc).lower()
        if any(token in msg for token in ("connect", "connection", "refused", "timed out", "timeout")):
            pytest.skip("Qdrant 미가동 (make up 필요)")
        raise
    if COLLECTION in {c.name for c in client.get_collections().collections}:
        client.delete_collection(COLLECTION)
    yield client
    client.delete_collection(COLLECTION)
    client.close()


@pytest.mark.asyncio
async def test_adapter_dense_finds_relevant(qclient, settings, monkeypatch):
    children = [_child(0, "manual"), _child(1, "incident"), _child(2, "manual")]
    await index_children(children, embedder=_DeterministicEmbedder(), client=qclient, settings=settings)

    # 어댑터/검색이 테스트 컬렉션·임베더를 쓰도록 모듈 심볼 monkeypatch
    monkeypatch.setattr(adapter, "get_embedder", lambda *a, **k: _DeterministicEmbedder())
    monkeypatch.setattr(adapter, "get_settings", lambda *a, **k: settings)
    monkeypatch.setattr(search_mod, "get_qdrant", lambda: qclient)
    monkeypatch.setattr(search_mod, "get_settings", lambda: settings)

    # 질문 = chunk0 본문과 동일 → 결정론 임베딩 cos=1 → pg0_000 최상위
    ids = await adapter.ranked_chunk_ids("manual 문서 본문 0", mode="dense", top_n=3)
    assert ids
    assert ids[0] == "pg0_000"


@pytest.mark.asyncio
async def test_adapter_domain_filter(qclient, settings, monkeypatch):
    children = [_child(0, "manual"), _child(1, "incident")]
    await index_children(children, embedder=_DeterministicEmbedder(), client=qclient, settings=settings)

    monkeypatch.setattr(adapter, "get_embedder", lambda *a, **k: _DeterministicEmbedder())
    monkeypatch.setattr(adapter, "get_settings", lambda *a, **k: settings)
    monkeypatch.setattr(search_mod, "get_qdrant", lambda: qclient)
    monkeypatch.setattr(search_mod, "get_settings", lambda: settings)

    ids = await adapter.ranked_chunk_ids("incident 문서 본문 1", mode="dense", domain="incident", top_n=5)
    assert ids == ["pg1_000"]  # 도메인 필터로 incident만
