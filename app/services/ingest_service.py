"""Confluence ingestion orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from app.db.confluence import ConfluenceClient
from app.rag.chunker import ChildChunk, MarkdownPage, ParentChunk, SemanticChunker
from app.rag.classifier import ChunkMetadataClassifier, DocumentProfileClassifier
from app.rag.cleaner import TextCleaner
from app.rag.masker import MarkdownMasker


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
        masker: MarkdownMasker | None = None,
        chunker: SemanticChunker | None = None,
        profile_classifier: DocumentProfileClassifier | None = None,
        metadata_classifier: ChunkMetadataClassifier | None = None,
    ) -> None:
        self.confluence = confluence or ConfluenceClient()
        self.cleaner = cleaner or TextCleaner()
        self.masker = masker or MarkdownMasker()
        self.chunker = chunker or SemanticChunker()
        self.profile_classifier = profile_classifier or DocumentProfileClassifier()
        self.metadata_classifier = metadata_classifier or ChunkMetadataClassifier()

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
        """Fetch recent pages, mask cleaned Markdown, and return semantic chunks."""

        cleaned_pages = await self.clean_recent_pages(hours=hours, limit=limit)
        return [self._chunk_cleaned_page(self._mask_page(page)) for page in cleaned_pages]

    async def prepare_recent_pages_for_embedding(self, hours: int = 24, limit: int = 50) -> list[ChunkedConfluencePage]:
        """Fetch, clean, mask, chunk, and classify recent pages before embedding."""

        cleaned_pages = await self.clean_recent_pages(hours=hours, limit=limit)
        prepared_pages: list[ChunkedConfluencePage] = []
        for page in cleaned_pages:
            masked_page = self._mask_page(page)
            chunking_profile = self.profile_classifier.classify_page(masked_page.title, masked_page.markdown)
            chunked_page = self._chunk_cleaned_page(masked_page)
            prepared_pages.append(
                ChunkedConfluencePage(
                    page=chunked_page.page,
                    parents=chunked_page.parents,
                    children=self.metadata_classifier.classify_batch(chunked_page.children, chunking_profile),
                )
            )

        return prepared_pages

    def _mask_page(self, page: CleanedConfluencePage) -> CleanedConfluencePage:
        return CleanedConfluencePage(
            page_id=page.page_id,
            title=page.title,
            space_key=page.space_key,
            markdown=self.masker.mask(page.markdown),
            html=page.html,
            last_modified=page.last_modified,
            version=page.version,
            url=page.url,
        )

    def _chunk_cleaned_page(self, page: CleanedConfluencePage) -> ChunkedConfluencePage:
        markdown_page = MarkdownPage(
            page_id=page.page_id,
            page_title=page.title,
            markdown=page.markdown,
            source_url=page.url,
            space_key=page.space_key,
            last_modified=page.last_modified,
        )
        parents, children = self.chunker.chunk(markdown_page)
        return ChunkedConfluencePage(page=page, parents=parents, children=children)
