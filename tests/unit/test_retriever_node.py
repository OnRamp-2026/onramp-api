import pytest

from app.agents.retriever import node as node_mod
from app.agents.retriever.node import retrieve_node
from app.agents.state import SourceDocument


def _hit(chunk_id, content, score, domain="장애대응"):
    payload = {
        "chunk_id": chunk_id,
        "content": content,
        "page_title": "제목",
        "source_url": "http://x",
        "space_key": "OnRamp",
        "domain": domain,
        "last_modified": "",
    }
    return type("SP", (), {"payload": payload, "score": score})()


class _FakeEmbedder:
    async def embed_query(self, text):
        return [0.1, 0.2, 0.3]


def _patch(monkeypatch, search_fn, rerank_obj):
    monkeypatch.setattr(node_mod, "get_embedder", lambda *a, **k: _FakeEmbedder())
    monkeypatch.setattr(node_mod, "dense_search", search_fn)
    monkeypatch.setattr(node_mod, "get_reranker", lambda *a, **k: rerank_obj)


@pytest.mark.asyncio
async def test_node_maps_to_source_document(monkeypatch):
    hits = [_hit("c1", "alpha", 0.9), _hit("c2", "beta", 0.8)]

    async def fake_search(qv, top_k, *, domain=None, **k):
        return hits

    class _R:
        def rerank(self, q, cands):
            return [(0.5, p) for _, p in cands]

    _patch(monkeypatch, fake_search, _R())
    out = await retrieve_node({"refined_query": "q", "domain": "장애대응"})
    docs = out["documents"]
    assert out["agent_trace"] == ["retriever"]
    assert all(isinstance(d, SourceDocument) for d in docs)
    assert docs[0].title == "제목"
    assert docs[0].content_snippet == "alpha"
    assert docs[0].score == 0.9


@pytest.mark.asyncio
async def test_node_domain_filter_fallback(monkeypatch):
    calls = []

    async def fake_search(qv, top_k, *, domain=None, **k):
        calls.append(domain)
        return [] if domain else [_hit("c1", "x", 0.7)]

    class _R:
        def rerank(self, q, cands):
            return [(0.1, p) for _, p in cands]

    _patch(monkeypatch, fake_search, _R())
    out = await retrieve_node({"refined_query": "q", "domain": "장애대응"})
    assert calls == ["장애대응", None]  # 필터→0건→무필터 재검색
    assert len(out["documents"]) == 1


@pytest.mark.asyncio
async def test_node_rerank_oom_fallback(monkeypatch):
    hits = [_hit("c1", "a", 0.3), _hit("c2", "b", 0.9)]

    async def fake_search(qv, top_k, *, domain=None, **k):
        return hits

    class _R:
        def rerank(self, q, cands):
            raise RuntimeError("OOM")

    _patch(monkeypatch, fake_search, _R())
    out = await retrieve_node({"refined_query": "q", "domain": "장애대응"})
    # 리랭커 실패 → vector score 순 폴백 (c2=0.9 먼저)
    assert out["documents"][0].content_snippet == "b"
    assert out["documents"][0].rerank_score == 0.0


@pytest.mark.asyncio
async def test_node_rerank_missing_dependency_fallback(monkeypatch):
    """sentence-transformers 미설치(ModuleNotFoundError)도 vector score 순 폴백."""
    hits = [_hit("c1", "a", 0.3), _hit("c2", "b", 0.9)]

    async def fake_search(qv, top_k, *, domain=None, **k):
        return hits

    class _R:
        def rerank(self, q, cands):
            raise ModuleNotFoundError("No module named 'sentence_transformers'")

    _patch(monkeypatch, fake_search, _R())
    out = await retrieve_node({"refined_query": "q", "domain": "장애대응"})
    assert out["documents"][0].content_snippet == "b"  # vec score 0.9 우선
    assert out["documents"][0].rerank_score == 0.0
