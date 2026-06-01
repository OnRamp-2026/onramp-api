"""Confluence ingestion orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from app.db.confluence import ConfluenceClient
from app.rag.chunker import ChildChunk, MarkdownPage, ParentChunk, SemanticChunker
from app.rag.cleaner import TextCleaner


@dataclass(frozen=True)
class CleanedConfluencePage:
    """A Confluence page after text cleaning."""

    page_id: str
    title: str
    space_key: str
    markdown: str
    html: str
    last_modified: str
    version: int | None
    url: str


@dataclass(frozen=True)
class ChunkedConfluencePage:
    """A cleaned Confluence page with generated parent and child chunks."""

    page: CleanedConfluencePage
    parents: list[ParentChunk]
    children: list[ChildChunk]


class IngestService:
    """Fetch changed Confluence pages and clean them for downstream RAG stages."""

    def __init__(
        self,
        confluence: ConfluenceClient | None = None,
        cleaner: TextCleaner | None = None,
        chunker: SemanticChunker | None = None,
    ) -> None:
        self.confluence = confluence or ConfluenceClient()
        self.cleaner = cleaner or TextCleaner()
        self.chunker = chunker or SemanticChunker()

    async def clean_recent_pages(self, hours: int = 24, limit: int = 50) -> list[CleanedConfluencePage]:
        """Fetch recently modified Confluence pages and return cleaned Markdown."""

        pages = await self.confluence.fetch_recent_pages(hours=hours, limit=limit)
        cleaned_pages: list[CleanedConfluencePage] = []

        for page in pages:
            cleaned_pages.append(
                CleanedConfluencePage(
                    page_id=page.page_id,
                    title=page.title,
                    space_key=page.space_key,
                    markdown=self.cleaner.clean(page.html),
                    html=page.html,
                    last_modified=page.last_modified,
                    version=page.version,
                    url=page.url,
                )
            )

        return cleaned_pages

    async def chunk_recent_pages(self, hours: int = 24, limit: int = 50) -> list[ChunkedConfluencePage]:
        """Fetch recently modified pages, clean them, and return semantic chunks."""

        cleaned_pages = await self.clean_recent_pages(hours=hours, limit=limit)
        chunked_pages: list[ChunkedConfluencePage] = []

        for page in cleaned_pages:
            markdown_page = MarkdownPage(
                page_id=page.page_id,
                page_title=page.title,
                markdown=page.markdown,
                source_url=page.url,
                space_key=page.space_key,
                last_modified=page.last_modified,
            )
            parents, children = self.chunker.chunk(markdown_page)
            chunked_pages.append(ChunkedConfluencePage(page=page, parents=parents, children=children))

        return chunked_pages
