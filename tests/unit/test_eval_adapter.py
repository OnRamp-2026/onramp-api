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
    # rerank 모드의 계보 조회(Qdrant facet) 차단 — 전부 계보 없음(version_fit 중립)
    monkeypatch.setattr(adapter, "get_lineages", lambda keys, **kw: {k: frozenset() for k in keys})


async def test_dense_mode_orders_by_vector_score(monkeypatch, patch_embedder):
    async def _search(qvec, top_k, *, domain=None, **kw):
        return [_point("c2", 0.8), _point("c1", 0.9), _point("c3", 0.7)]

    monkeypatch.setattr(search_mod, "dense_search", _search)
    ids = await adapter.ranked_chunk_ids("q", mode="dense", top_n=2, settings=Settings())
    assert ids == ["c1", "c2"]  # score 내림차순, top_n=2


async def test_dense_mode_applies_domain_bonus(monkeypatch, patch_embedder):
    """soft 재현 — dense 모드도 도메인 가산을 정렬 키에 반영(운영 _vector_fallback과 동일)."""

    def _dp(chunk_id, score, domain):
        return SimpleNamespace(
            id=chunk_id, score=score, payload={"chunk_id": chunk_id, "content": "c", "domain": domain}
        )

    async def _search(qvec, top_k, *, domain=None, **kw):
        # vector score는 c1 > c2지만, c2만 router domain(manual)과 일치 → 가산 후 c2가 앞서야 함
        return [_dp("c1", 0.50, "incident"), _dp("c2", 0.45, "manual")]

    monkeypatch.setattr(search_mod, "dense_search", _search)
    s = Settings()  # domain_match_weight=0.1 → 0.45+0.1=0.55 > 0.50
    ids = await adapter.ranked_chunk_ids("q", mode="dense", domains=["manual"], filter_mode="soft", settings=s)
    assert ids == ["c2", "c1"]  # 도메인 가산이 순서를 뒤집음
    # 가산이 없으면(domain=None) 순수 vector score 순
    ids_no_domain = await adapter.ranked_chunk_ids("q", mode="dense", domains=None, filter_mode="soft", settings=s)
    assert ids_no_domain == ["c1", "c2"]


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
    # 점수 분리(#103): top_score는 부스트 합산(ranking), tau_score는 top_n 내 최대 raw
    assert result.tau_score == pytest.approx(0.9)
    assert result.raw_scores == pytest.approx((0.9, 0.5))
    assert result.top_score >= 0.9  # ranking ≥ raw (가산식 부스트)
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
        "q", mode="dense", domains=["incident"], filter_mode="hybrid", settings=Settings()
    )
    assert ids == ["c1"]
    assert calls == ["incident", None]  # 무필터 재검색 발생 (필터용 domain=domains[0])


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
    # τ 비교는 tau_score(rerank 모드=raw) 기준 (#103) — top_score(부스트 합산)가 아니라
    r = RetrievalResult(chunk_ids=["c1"], top_score=0.7, n=1, tau_score=0.5)
    assert predicted_answerable(r, floor=0.4, min_docs=1) is True
    assert predicted_answerable(r, floor=0.6, min_docs=1) is False  # raw 미달 (top_score 0.7이어도)
    empty = RetrievalResult(chunk_ids=[], top_score=0.0, n=0, tau_score=0.0)
    assert predicted_answerable(empty, floor=0.0, min_docs=1) is False  # 문서 0
