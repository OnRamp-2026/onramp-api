from app.db.confluence import ConfluencePage
from app.rag.classifier import KOREAN_DOMAIN_MAP
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

    async def fetch_all_pages(self, limit: int) -> list[ConfluencePage]:
        return await self.fetch_recent_pages(hours=0, limit=limit)


class FakeControlConfluenceClient:
    async def fetch_all_pages(self, limit: int) -> list[ConfluencePage]:
        return await self.fetch_recent_pages(hours=0, limit=limit)

    async def fetch_recent_pages(self, hours: int, limit: int) -> list[ConfluencePage]:
        return [
            ConfluencePage(
                page_id="456",
                title="정책 회의록",
                space_key="OnRamp",
                html="""
                <main>
                  <h1>정책 회의록</h1>
                  <h2>결정사항</h2>
                  <p>Qdrant collection 정책을 확정한다.</p>
                  <h2>액션아이템</h2>
                  <ul>
                    <li>담당자: 플랫폼팀</li>
                    <li>기한: 2026-06-03</li>
                  </ul>
                  <h2>리스크</h2>
                  <p>블로커: 문서가 길어지면 결정 맥락이 잘릴 수 있다.</p>
                </main>
                """,
                last_modified="2026-06-01T12:35:00.000+0900",
                version=3,
                url="https://example.atlassian.net/wiki/spaces/OnRamp/pages/456/Policy+Meeting",
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
    joined_content = "\n".join(child.content for child in pages[0].children)
    assert "admin@example.com" not in joined_content
    assert "[MASKED_EMAIL]" in joined_content


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


async def test_prepare_recent_pages_for_embedding_uses_control_chunker_for_control_like_pages() -> None:
    service = IngestService(confluence=FakeControlConfluenceClient())  # type: ignore[arg-type]

    pages = await service.prepare_recent_pages_for_embedding(hours=24, limit=10)

    assert len(pages) == 1
    assert {parent.section_type for parent in pages[0].parents} >= {"decision", "action_item", "risk"}
    assert all(child.chunking_profile == "control_like" for child in pages[0].children)
    assert any(child.section_type == "action_item" for child in pages[0].children)
    assert "청킹 프로필: control_like" in "\n".join(child.embedding_text for child in pages[0].children)


async def test_clean_all_pages_fetches_and_cleans_confluence_pages() -> None:
    service = IngestService(confluence=FakeConfluenceClient())  # type: ignore[arg-type]

    pages = await service.clean_all_pages(limit=10)

    assert len(pages) == 1
    assert pages[0].page_id == "123"
    assert "# API Runbook" in pages[0].markdown
    assert "Restart pod." in pages[0].markdown
    assert "Noise" not in pages[0].markdown


async def test_chunk_all_pages_fetches_cleans_and_chunks_confluence_pages() -> None:
    service = IngestService(confluence=FakeConfluenceClient())  # type: ignore[arg-type]

    pages = await service.chunk_all_pages(limit=10)

    assert len(pages) == 1
    assert pages[0].page.page_id == "123"
    assert pages[0].children
    joined_content = "\n".join(child.content for child in pages[0].children)
    assert "admin@example.com" not in joined_content
    assert "[MASKED_EMAIL]" in joined_content


async def test_prepare_children_inherit_parent_domain() -> None:
    # prepare 경로에서 각 child는 소속 parent의 domain(영문 정규화)을 상속한다 (#51)
    service = IngestService(confluence=FakeControlConfluenceClient())  # type: ignore[arg-type]

    pages = await service.prepare_recent_pages_for_embedding(hours=24, limit=10)

    page = pages[0]
    parent_domain = {p.parent_id: KOREAN_DOMAIN_MAP.get(p.domain, p.domain) for p in page.parents}
    assert page.children
    for child in page.children:
        assert child.domain == parent_domain[child.parent_id]


async def test_prepare_all_pages_for_embedding_masks_and_classifies_chunks() -> None:
    service = IngestService(confluence=FakeConfluenceClient())  # type: ignore[arg-type]

    pages = await service.prepare_all_pages_for_embedding(limit=10)

    assert len(pages) == 1
    embedding_text = "\n".join(child.embedding_text for child in pages[0].children)
    assert "abc.def.ghi1234567890" not in embedding_text
    assert "[MASKED_TOKEN]" in embedding_text
    assert all(child.tags for child in pages[0].children)
