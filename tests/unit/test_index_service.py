from app.rag.chunker import ChildChunk
from app.services.index_service import IndexService
from app.services.ingest_service import ChunkedConfluencePage, CleanedConfluencePage


def _child(chunk_id: str) -> ChildChunk:
    return ChildChunk(
        chunk_id=chunk_id,
        parent_id="p",
        page_id="pg",
        page_title="문서",
        content="본문",
        embedding_text=f"embedding text {chunk_id}",
        heading_path=["문서"],
        chunk_index=0,
        token_count=10,
        overlap_from_previous=0,
        source_url="https://example.atlassian.net/wiki/spaces/OnRamp/pages/pg",
        space_key="OnRamp",
        last_modified="2026-06-01T00:00:00.000+0900",
        hash="hash",
        chunking_profile="runbook_like",
        domain="manual",
    )


def _page(page_id: str, children: list[ChildChunk]) -> ChunkedConfluencePage:
    return ChunkedConfluencePage(
        page=CleanedConfluencePage(
            page_id=page_id,
            title="문서",
            space_key="OnRamp",
            markdown="",
            html="",
            last_modified="2026-06-01T00:00:00.000+0900",
            version=1,
            url="",
        ),
        parents=[],
        children=children,
    )


class _FakeIngestService:
    def __init__(self, pages: list[ChunkedConfluencePage]) -> None:
        self.pages = pages
        self.calls: list[tuple[int, int]] = []
        self.all_calls: list[int] = []

    async def prepare_recent_pages_for_embedding(self, hours: int, limit: int) -> list[ChunkedConfluencePage]:
        self.calls.append((hours, limit))
        return self.pages

    async def prepare_all_pages_for_embedding(self, limit: int) -> list[ChunkedConfluencePage]:
        self.all_calls.append(limit)
        return self.pages


async def test_index_recent_pages_flattens_children_and_returns_summary() -> None:
    captured_children: list[ChildChunk] = []

    async def fake_index_children(children: list[ChildChunk]) -> int:
        captured_children.extend(children)
        return len(children)

    ingest = _FakeIngestService(
        [
            _page("p1", [_child("p1_000"), _child("p1_001")]),
            _page("p2", [_child("p2_000")]),
        ]
    )
    service = IndexService(ingest_service=ingest, index_children_fn=fake_index_children)  # type: ignore[arg-type]

    result = await service.index_recent_pages(hours=12, limit=7)

    assert ingest.calls == [(12, 7)]
    assert [child.chunk_id for child in captured_children] == ["p1_000", "p1_001", "p2_000"]
    assert result.pages_indexed == 2
    assert result.chunks_indexed == 3


async def test_index_all_pages_uses_full_prepare_path() -> None:
    captured_children: list[ChildChunk] = []

    async def fake_index_children(children: list[ChildChunk]) -> int:
        captured_children.extend(children)
        return len(children)

    ingest = _FakeIngestService([_page("p1", [_child("p1_000"), _child("p1_001")])])
    service = IndexService(ingest_service=ingest, index_children_fn=fake_index_children)  # type: ignore[arg-type]

    result = await service.index_all_pages(limit=9)

    assert ingest.all_calls == [9]
    assert ingest.calls == []  # recent 경로는 타지 않음
    assert [child.chunk_id for child in captured_children] == ["p1_000", "p1_001"]
    assert result.pages_indexed == 1
    assert result.chunks_indexed == 2


async def test_index_recent_pages_handles_pages_without_children() -> None:
    async def fake_index_children(children: list[ChildChunk]) -> int:
        assert children == []
        return 0

    service = IndexService(
        ingest_service=_FakeIngestService([_page("empty", [])]),  # type: ignore[arg-type]
        index_children_fn=fake_index_children,
    )

    result = await service.index_recent_pages()

    assert result.pages_indexed == 1
    assert result.chunks_indexed == 0
