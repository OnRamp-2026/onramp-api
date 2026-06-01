from app.db.confluence import ConfluencePage
from app.services.ingest_service import IngestService


class FakeConfluenceClient:
    async def fetch_recent_pages(self, hours: int, limit: int) -> list[ConfluencePage]:
        return [
            ConfluencePage(
                page_id="123",
                title="API Runbook",
                space_key="OnRamp",
                html="""
                <main>
                  <nav>Noise</nav>
                  <h1>API Runbook</h1>
                  <p>Restart pod.</p>
                  <p>admin@example.com</p>
                  <p>Authorization: Bearer abc.def.ghi1234567890</p>
                </main>
                """,
                last_modified="2026-05-28T12:35:00.000+0900",
                version=7,
                url="https://example.atlassian.net/wiki/spaces/OnRamp/pages/123/API+Runbook",
            )
        ]


async def test_clean_recent_pages_fetches_and_cleans_confluence_pages() -> None:
    service = IngestService(confluence=FakeConfluenceClient())  # type: ignore[arg-type]

    pages = await service.clean_recent_pages(hours=1, limit=10)

    assert len(pages) == 1
    assert pages[0].page_id == "123"
    assert "# API Runbook" in pages[0].markdown
    assert "Restart pod." in pages[0].markdown
    assert "Noise" not in pages[0].markdown
    assert "admin@example.com" in pages[0].markdown


async def test_chunk_recent_pages_fetches_cleans_and_chunks_confluence_pages() -> None:
    service = IngestService(confluence=FakeConfluenceClient())  # type: ignore[arg-type]

    pages = await service.chunk_recent_pages(hours=24, limit=10)

    assert len(pages) == 1
    assert pages[0].page.page_id == "123"
    assert pages[0].parents
    assert pages[0].children
    assert pages[0].children[0].page_id == "123"
    assert pages[0].children[0].parent_id == pages[0].parents[0].parent_id
    assert "문서: API Runbook" in pages[0].children[0].embedding_text


async def test_prepare_recent_pages_for_embedding_masks_and_classifies_chunks() -> None:
    service = IngestService(confluence=FakeConfluenceClient())  # type: ignore[arg-type]

    pages = await service.prepare_recent_pages_for_embedding(hours=24, limit=10)

    assert len(pages) == 1
    content = "\n".join(child.content for child in pages[0].children)
    embedding_text = "\n".join(child.embedding_text for child in pages[0].children)
    assert "admin@example.com" not in content
    assert "abc.def.ghi1234567890" not in embedding_text
    assert "[MASKED_EMAIL]" in content
    assert "[MASKED_TOKEN]" in embedding_text
    assert all(child.chunking_profile == "runbook_like" for child in pages[0].children)
    assert "청킹 프로필: runbook_like" in embedding_text
    assert all(child.tags for child in pages[0].children)
    assert all(
        child.domain in {"incident", "manual", "api_reference", "meeting_note", "planning"}
        for child in pages[0].children
    )
