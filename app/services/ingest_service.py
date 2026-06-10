"""Confluence ingestion orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from app.db.confluence import ConfluenceClient, ConfluencePage
from app.rag.chunker import ChildChunk, ControlDocChunker, MarkdownPage, ParentChunk, SemanticChunker
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
        control_chunker: ControlDocChunker | None = None,
        profile_classifier: DocumentProfileClassifier | None = None,
        metadata_classifier: ChunkMetadataClassifier | None = None,
    ) -> None:
        self.confluence = confluence or ConfluenceClient()
        self.cleaner = cleaner or TextCleaner()
        self.masker = masker or MarkdownMasker()
        self.chunker = chunker or SemanticChunker()
        self.control_chunker = control_chunker or ControlDocChunker()
        self.profile_classifier = profile_classifier or DocumentProfileClassifier()
        self.metadata_classifier = metadata_classifier or ChunkMetadataClassifier()

    # ── recent (증분, lastmodified 기준) ──────────────────────────────────
    async def clean_recent_pages(self, hours: int = 24, limit: int = 50) -> list[CleanedConfluencePage]:
        """Fetch recently modified Confluence pages and return cleaned Markdown."""

        pages = await self.confluence.fetch_recent_pages(hours=hours, limit=limit)
        return self._clean(pages)

    async def chunk_recent_pages(self, hours: int = 24, limit: int = 50) -> list[ChunkedConfluencePage]:
        """Fetch recent pages, mask cleaned Markdown, and return semantic chunks."""

        return self._chunk(await self.clean_recent_pages(hours=hours, limit=limit))

    async def prepare_recent_pages_for_embedding(self, hours: int = 24, limit: int = 50) -> list[ChunkedConfluencePage]:
        """Fetch, clean, mask, chunk, and classify recent pages before embedding."""

        return self._prepare(await self.clean_recent_pages(hours=hours, limit=limit))

    # ── all (전체 적재, lastmodified 무시) ────────────────────────────────
    async def clean_all_pages(self, limit: int = 50) -> list[CleanedConfluencePage]:
        """Fetch every page in the space (initial full load) and return cleaned Markdown."""

        pages = await self.confluence.fetch_all_pages(limit=limit)
        return self._clean(pages)

    async def chunk_all_pages(self, limit: int = 50) -> list[ChunkedConfluencePage]:
        """Fetch all pages, mask cleaned Markdown, and return semantic chunks."""

        return self._chunk(await self.clean_all_pages(limit=limit))

    async def prepare_all_pages_for_embedding(self, limit: int = 50) -> list[ChunkedConfluencePage]:
        """Fetch, clean, mask, chunk, and classify all pages before embedding."""

        return self._prepare(await self.clean_all_pages(limit=limit))

    async def masked_all_pages(self, limit: int = 50) -> list[CleanedConfluencePage]:
        """전체 페이지를 cleaned+masked 상태로 반환 (문서 도메인 분류 dry-run 입력용).

        마스킹 로직은 기존 _mask_page를 그대로 재사용 — 분류 단계에 마스킹되지 않은 원문이 흘러가지 않게 한다.
        """
        return [self._mask_page(page) for page in await self.clean_all_pages(limit=limit)]

    # ── recent/all 공통 변환부 (fetch만 다르고 이하 동일) ─────────────────
    def _clean(self, pages: list[ConfluencePage]) -> list[CleanedConfluencePage]:
        return [
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
            for page in pages
        ]

    def _chunk(self, cleaned_pages: list[CleanedConfluencePage]) -> list[ChunkedConfluencePage]:
        return [self._chunk_cleaned_page(self._mask_page(page)) for page in cleaned_pages]

    def _prepare(self, cleaned_pages: list[CleanedConfluencePage]) -> list[ChunkedConfluencePage]:
        prepared_pages: list[ChunkedConfluencePage] = []
        for page in cleaned_pages:
            masked_page = self._mask_page(page)
            chunking_profile = self.profile_classifier.classify_page(masked_page.title, masked_page.markdown)
            chunked_page = self._chunk_cleaned_page(masked_page, chunking_profile=chunking_profile)
            # child가 자기 추론이 아니라 소속 parent의 domain을 상속하도록 parent domain 맵 전달
            parent_domains = {parent.parent_id: parent.domain for parent in chunked_page.parents}
            prepared_pages.append(
                ChunkedConfluencePage(
                    page=chunked_page.page,
                    parents=chunked_page.parents,
                    children=self.metadata_classifier.classify_batch(
                        chunked_page.children, chunking_profile, parent_domains
                    ),
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

    def _chunk_cleaned_page(
        self, page: CleanedConfluencePage, chunking_profile: str = "runbook_like"
    ) -> ChunkedConfluencePage:
        markdown_page = MarkdownPage(
            page_id=page.page_id,
            page_title=page.title,
            markdown=page.markdown,
            source_url=page.url,
            space_key=page.space_key,
            last_modified=page.last_modified,
        )
        chunker = self.control_chunker if chunking_profile == "control_like" else self.chunker
        parents, children = chunker.chunk(markdown_page)
        return ChunkedConfluencePage(page=page, parents=parents, children=children)
