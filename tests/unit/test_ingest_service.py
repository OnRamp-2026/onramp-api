from app.db.confluence import ConfluencePage
from app.services.ingest_service import IngestService


class FakeConfluenceClient:
    async def fetch_recent_pages(self, hours: int, limit: int) -> list[ConfluencePage]:
        return [
            ConfluencePage(
                page_id="123",
                title="API Runbook",
                space_key="TRUSTRAG",
                html="""
                <main>
                  <nav>Noise</nav>
                  <h1>API Runbook</h1>
                  <p>Restart pod.</p>
                </main>
                """,
                last_modified="2026-05-28T12:35:00.000+0900",
                version=7,
                url="https://example.atlassian.net/wiki/spaces/TRUSTRAG/pages/123/API+Runbook",
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
