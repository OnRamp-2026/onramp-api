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

    async def fake(qv, top_k, *, domain=None, settings=None, **kw):
        calls.append(domain)
        return [_pt("a", 0.2)]  # 저품질이어도 hard는 확장 안 함

    monkeypatch.setattr(search_mod, "dense_search", fake)
    out = await search_with_mode([0.1], 5, domain="manual", mode="hard", settings=Settings())
    assert [p.id for p in out] == ["a"]
    assert calls == ["manual"]


@pytest.mark.asyncio
async def test_search_with_mode_soft_ignores_domain(monkeypatch):
    calls = []

    async def fake(qv, top_k, *, domain=None, settings=None, **kw):
        calls.append(domain)
        return [_pt("a", 0.9)]

    monkeypatch.setattr(search_mod, "dense_search", fake)
    await search_with_mode([0.1], 5, domain="manual", mode="soft", settings=Settings())
    assert calls == [None]  # 무필터


@pytest.mark.asyncio
async def test_search_with_mode_hybrid_expands_on_low_quality(monkeypatch):
    calls = []

    async def fake(qv, top_k, *, domain=None, settings=None, **kw):
        calls.append(domain)
        return [_pt("a", 0.2)] if domain else [_pt("a", 0.2), _pt("b", 0.9)]

    monkeypatch.setattr(search_mod, "dense_search", fake)
    out = await search_with_mode([0.1], 5, domain="manual", mode="hybrid", settings=Settings())
    assert calls == ["manual", None]  # 저품질 → 무필터 확장
    assert {p.id for p in out} == {"a", "b"}


@pytest.mark.asyncio
async def test_search_with_mode_hybrid_no_expand_when_high_quality(monkeypatch):
    calls = []

    async def fake(qv, top_k, *, domain=None, settings=None, **kw):
        calls.append(domain)
        return [_pt("a", 0.9)]

    monkeypatch.setattr(search_mod, "dense_search", fake)
    await search_with_mode([0.1], 5, domain="manual", mode="hybrid", settings=Settings())
    assert calls == ["manual"]  # 고품질 → 확장 없음


# ── 재검색 사다리 필터 (#108) ────────────────────────────────────────


def test_build_filter_composes_ladder_conditions():
    from app.agents.retriever.search import SearchFilters, _build_filter

    f = _build_filter(
        "manual",
        SearchFilters(version="2.4", pinned_doc_keys=("apache:mpm",), excluded_doc_keys=("apache:done",)),
    )
    musts = {c.key for c in f.must}
    assert musts == {"domain", "product_version", "doc_key"}
    assert [c.key for c in f.must_not] == ["doc_key"]


def test_build_filter_none_when_empty():
    from app.agents.retriever.search import SearchFilters, _build_filter

    assert _build_filter(None, None) is None
    assert _build_filter(None, SearchFilters()) is None


def test_build_filter_includes_tenant_and_source():
    from app.agents.retriever.search import _build_filter

    result = _build_filter(None, None, tenant_id="tenant-a", source="github")
    assert {condition.key for condition in result.must} == {"tenant_id", "source"}


async def test_soft_mode_still_applies_ladder_filters(monkeypatch):
    """domain은 soft(무필터)여도 사다리 필터는 항상 적용된다 — 사다리 전략의 본질."""
    from app.agents.retriever.search import SearchFilters

    seen = {}

    async def fake(qv, top_k, *, domain=None, settings=None, filters=None, **kw):
        seen["domain"] = domain
        seen["filters"] = filters
        return []

    monkeypatch.setattr(search_mod, "dense_search", fake)
    filters = SearchFilters(version="2.4")
    await search_with_mode([0.1], 5, domain="manual", mode="soft", filters=filters, settings=Settings())
    assert seen["domain"] is None  # soft → 도메인 무필터
    assert seen["filters"] is filters  # 사다리 필터는 유지


@pytest.mark.asyncio
async def test_hybrid_search_gate_uses_query_text(monkeypatch):
    seen = {}

    async def fake_hybrid(query_text, query_vector, *, domain, filters, settings, dense_search_fn, **kwargs):  # noqa: ANN202, ARG001
        seen["query_text"] = query_text
        seen["domain"] = domain
        return [_pt("hybrid", 0.1)]

    monkeypatch.setattr("app.rag.hybrid_search.hybrid_search", fake_hybrid)
    settings = Settings(hybrid_search_enabled=True)

    out = await search_with_mode([0.1], 5, domain="manual", mode="hard", query_text="장애 대응", settings=settings)

    assert [p.id for p in out] == ["hybrid"]
    assert seen == {"query_text": "장애 대응", "domain": "manual"}


@pytest.mark.asyncio
async def test_hybrid_search_gate_respects_soft_domain(monkeypatch):
    seen = {}

    async def fake_hybrid(query_text, query_vector, *, domain, filters, settings, dense_search_fn, **kwargs):  # noqa: ANN202, ARG001
        seen["domain"] = domain
        return []

    monkeypatch.setattr("app.rag.hybrid_search.hybrid_search", fake_hybrid)

    await search_with_mode(
        [0.1],
        5,
        domain="manual",
        mode="soft",
        query_text="장애 대응",
        settings=Settings(hybrid_search_enabled=True),
    )

    assert seen["domain"] is None
