import pytest

from app.agents.retriever.search import dense_search
from app.config import Settings


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
