from app.services.ingest_service import ChunkedConfluencePage, CleanedConfluencePage
from scripts.prepare_recent_confluence_pages import _safe_stem


def test_safe_stem_truncates_long_titles() -> None:
    page = ChunkedConfluencePage(
        page=CleanedConfluencePage(
            page_id="123",
            title="긴 제목 " * 80,
            space_key="OnRamp",
            markdown="",
            html="",
            last_modified="",
            version=1,
            url="",
        ),
        parents=[],
        children=[],
    )

    stem = _safe_stem(page)

    assert stem.startswith("123-")
    assert len(stem) <= 124
