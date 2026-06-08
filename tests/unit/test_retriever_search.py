import pytest

from app.agents.retriever import search as search_mod
from app.agents.retriever.search import dense_search, search_with_mode
from app.config import Settings


def _pt(pid, score):
    return type("SP", (), {"id": pid, "score": score, "payload": {"chunk_id": pid}})()


class _FakeResp:
    def __init__(self, points):
        self.points = points


class _FakeClient:
    def __init__(self):
        self.last_kwargs = None

    def query_points(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResp(["p1", "p2"])


@pytest.mark.asyncio
async def test_dense_search_no_filter_when_domain_none():
    client = _FakeClient()
    points = await dense_search([0.1, 0.2], 5, client=client, settings=Settings())
    assert points == ["p1", "p2"]
    assert client.last_kwargs["query_filter"] is None
    assert client.last_kwargs["limit"] == 5


@pytest.mark.asyncio
async def test_dense_search_builds_domain_filter():
    client = _FakeClient()
    await dense_search([0.1], 5, domain="장애대응", client=client, settings=Settings())
    flt = client.last_kwargs["query_filter"]
    assert flt is not None
    assert flt.must[0].key == "domain"
    assert flt.must[0].match.value == "장애대응"


@pytest.mark.asyncio
async def test_search_with_mode_hard_no_expansion(monkeypatch):
    calls = []

    async def fake(qv, top_k, *, domain=None, settings=None):
        calls.append(domain)
        return [_pt("a", 0.2)]  # 저품질이어도 hard는 확장 안 함

    monkeypatch.setattr(search_mod, "dense_search", fake)
    out = await search_with_mode([0.1], 5, domain="manual", mode="hard", settings=Settings())
    assert [p.id for p in out] == ["a"]
    assert calls == ["manual"]


@pytest.mark.asyncio
async def test_search_with_mode_soft_ignores_domain(monkeypatch):
    calls = []

    async def fake(qv, top_k, *, domain=None, settings=None):
        calls.append(domain)
        return [_pt("a", 0.9)]

    monkeypatch.setattr(search_mod, "dense_search", fake)
    await search_with_mode([0.1], 5, domain="manual", mode="soft", settings=Settings())
    assert calls == [None]  # 무필터


@pytest.mark.asyncio
async def test_search_with_mode_hybrid_expands_on_low_quality(monkeypatch):
    calls = []

    async def fake(qv, top_k, *, domain=None, settings=None):
        calls.append(domain)
        return [_pt("a", 0.2)] if domain else [_pt("a", 0.2), _pt("b", 0.9)]

    monkeypatch.setattr(search_mod, "dense_search", fake)
    out = await search_with_mode([0.1], 5, domain="manual", mode="hybrid", settings=Settings())
    assert calls == ["manual", None]  # 저품질 → 무필터 확장
    assert {p.id for p in out} == {"a", "b"}


@pytest.mark.asyncio
async def test_search_with_mode_hybrid_no_expand_when_high_quality(monkeypatch):
    calls = []

    async def fake(qv, top_k, *, domain=None, settings=None):
        calls.append(domain)
        return [_pt("a", 0.9)]

    monkeypatch.setattr(search_mod, "dense_search", fake)
    await search_with_mode([0.1], 5, domain="manual", mode="hybrid", settings=Settings())
    assert calls == ["manual"]  # 고품질 → 확장 없음
