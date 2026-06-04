import scripts.index_recent_confluence_pages as cli
from app.services.index_service import IndexResult


class _FakeIndexService:
    def __init__(self) -> None:
        self.recent_calls: list[tuple[int, int]] = []
        self.all_calls: list[int] = []

    async def index_recent_pages(self, hours: int, limit: int) -> IndexResult:
        self.recent_calls.append((hours, limit))
        return IndexResult(pages_indexed=0, chunks_indexed=0)

    async def index_all_pages(self, limit: int) -> IndexResult:
        self.all_calls.append(limit)
        return IndexResult(pages_indexed=0, chunks_indexed=0)


async def test_run_all_routes_to_index_all_pages(monkeypatch) -> None:
    fake = _FakeIndexService()
    monkeypatch.setattr(cli, "IndexService", lambda: fake)

    await cli.run(hours=24, limit=5, all_pages=True)

    assert fake.all_calls == [5]
    assert fake.recent_calls == []


async def test_run_default_routes_to_index_recent_pages(monkeypatch) -> None:
    fake = _FakeIndexService()
    monkeypatch.setattr(cli, "IndexService", lambda: fake)

    await cli.run(hours=12, limit=7)

    assert fake.recent_calls == [(12, 7)]
    assert fake.all_calls == []
