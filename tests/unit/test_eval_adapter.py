"""검색 어댑터 단위 테스트 (embedder/search/reranker monkeypatch — Qdrant/OpenAI 불필요).

패턴: tests/unit/test_retriever_node.py 와 동일하게 모듈 심볼을 monkeypatch.
"""

from types import SimpleNamespace

import pytest

from app.agents.retriever import search as search_mod
from app.config import Settings
from app.eval import retrieval_adapter as adapter
from app.eval.retrieval_adapter import RetrievalResult, predicted_answerable


def _point(chunk_id: str, score: float, content: str = "내용"):
    # id: search_with_mode의 merge(dedupe)가 point.id를 사용
    return SimpleNamespace(id=chunk_id, score=score, payload={"chunk_id": chunk_id, "content": content})


class _Embedder:
    async def embed_query(self, text: str):
        return [0.1, 0.2, 0.3]


@pytest.fixture
def patch_embedder(monkeypatch):
    monkeypatch.setattr(adapter, "get_embedder", lambda *a, **k: _Embedder())


async def test_dense_mode_orders_by_vector_score(monkeypatch, patch_embedder):
    async def _search(qvec, top_k, *, domain=None, **kw):
        return [_point("c2", 0.8), _point("c1", 0.9), _point("c3", 0.7)]

    monkeypatch.setattr(search_mod, "dense_search", _search)
    ids = await adapter.ranked_chunk_ids("q", mode="dense", top_n=2, settings=Settings())
    assert ids == ["c1", "c2"]  # score 내림차순, top_n=2


async def test_rerank_mode_reorders(monkeypatch, patch_embedder):
    async def _search(qvec, top_k, *, domain=None, **kw):
        return [_point("c1", 0.9), _point("c2", 0.8), _point("c3", 0.7)]

    class _Reranker:
        def rerank(self, query, candidates):
            # 입력 순서(c1,c2,c3)에 rerank 점수 부여 → c2 최고
            scores = [0.1, 0.9, 0.5]
            return [(s, payload) for s, (_, payload) in zip(scores, candidates, strict=True)]

    monkeypatch.setattr(search_mod, "dense_search", _search)
    monkeypatch.setattr(adapter, "get_reranker", lambda *a, **k: _Reranker())

    result = await adapter.retrieve_for_eval("q", mode="rerank", top_n=2, settings=Settings())
    assert result.chunk_ids == ["c2", "c3"]  # rerank 점수 내림차순
    assert result.top_score == pytest.approx(0.9)
    assert result.n == 2


async def test_domain_overfilter_falls_back(monkeypatch, patch_embedder):
    calls: list[str | None] = []

    async def _search(qvec, top_k, *, domain=None, **kw):
        calls.append(domain)
        if domain is not None:
            return []  # 도메인 과필터 0건
        return [_point("c1", 0.9)]

    monkeypatch.setattr(search_mod, "dense_search", _search)
    ids = await adapter.ranked_chunk_ids(
        "q", mode="dense", domain="incident", filter_mode="hybrid", settings=Settings()
    )
    assert ids == ["c1"]
    assert calls == ["incident", None]  # 무필터 재검색 발생


async def test_reranker_failure_falls_back_to_vector(monkeypatch, patch_embedder):
    async def _search(qvec, top_k, *, domain=None, **kw):
        return [_point("c1", 0.9), _point("c2", 0.8)]

    class _BoomReranker:
        def rerank(self, query, candidates):
            raise RuntimeError("OOM")

    monkeypatch.setattr(search_mod, "dense_search", _search)
    monkeypatch.setattr(adapter, "get_reranker", lambda *a, **k: _BoomReranker())
    ids = await adapter.ranked_chunk_ids("q", mode="rerank", settings=Settings())
    assert ids == ["c1", "c2"]  # vector score 순 폴백


async def test_non_positive_top_raises(patch_embedder) -> None:
    with pytest.raises(ValueError, match="1 이상"):
        await adapter.retrieve_for_eval("q", mode="dense", top_n=0, settings=Settings())


def test_predicted_answerable() -> None:
    r = RetrievalResult(chunk_ids=["c1"], top_score=0.5, n=1)
    assert predicted_answerable(r, floor=0.4, min_docs=1) is True
    assert predicted_answerable(r, floor=0.6, min_docs=1) is False  # 점수 미달
    empty = RetrievalResult(chunk_ids=[], top_score=0.0, n=0)
    assert predicted_answerable(empty, floor=0.0, min_docs=1) is False  # 문서 0
