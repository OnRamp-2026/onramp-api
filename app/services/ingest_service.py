"""Confluence ingestion orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.config import Settings, get_settings
from app.db.confluence import ConfluenceClient, ConfluencePage
from app.rag.chunker import ChildChunk, ControlDocChunker, MarkdownPage, ParentChunk, SemanticChunker
from app.rag.classifier import ChunkMetadataClassifier, DocumentProfileClassifier
from app.rag.cleaner import TextCleaner
from app.rag.labels import is_eol, make_doc_key, parse_product_version, parse_site
from app.rag.llm_classifier import DocumentDomainClassifier, DomainResult
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
    # 버전 계보 메타 (#94 — 라벨 파생, ChildChunk까지 관통해 Qdrant payload가 된다)
    site: str = ""
    product_version: str = ""
    doc_key: str = ""
    is_eol: bool = False


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
        domain_classifier: DocumentDomainClassifier | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.confluence = confluence or ConfluenceClient()
        self.cleaner = cleaner or TextCleaner()
        self.masker = masker or MarkdownMasker()
        self.chunker = chunker or SemanticChunker()
        self.control_chunker = control_chunker or ControlDocChunker()
        self.profile_classifier = profile_classifier or DocumentProfileClassifier()
        self.metadata_classifier = metadata_classifier or ChunkMetadataClassifier()
        self.domain_classifier = domain_classifier or DocumentDomainClassifier()
        self.settings = settings or get_settings()

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

        return await self._prepare(await self.clean_recent_pages(hours=hours, limit=limit))

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

        return await self._prepare(await self.clean_all_pages(limit=limit))

    # ── GitHub 소스 (이미 Markdown — clean 불필요, 동일한 mask→chunk→classify 재사용) ──
    async def prepare_github_pages(self, pages: list[MarkdownPage]) -> list[ChunkedConfluencePage]:
        """GitHub MarkdownPage(README·docs·이슈/PR)를 confluence와 동일한 파이프라인으로 준비.

        GitHub 원문은 이미 Markdown이라 HTML cleaning을 건너뛰고, mask→profile→chunk→metadata
        분류는 그대로 재사용한다. ``html``에는 원문 Markdown을 보존(원장 raw_html).
        """
        return await self._prepare([self._github_to_cleaned(page) for page in pages])

    def _github_to_cleaned(self, page: MarkdownPage) -> CleanedConfluencePage:
        return CleanedConfluencePage(
            page_id=page.page_id,  # gh:repo:path | gh:repo#number
            title=page.page_title,
            space_key=page.space_key,  # repo 이름
            markdown=page.markdown,
            html=page.markdown,  # GitHub 원문 = Markdown
            last_modified=page.last_modified,
            version=None,
            url=page.source_url,
            site=page.site,
            product_version=page.product_version,
            doc_key=page.doc_key or make_doc_key(page.site, page.page_title),
            is_eol=page.is_eol,
        )

    # ── recent/all 공통 변환부 (fetch만 다르고 이하 동일) ─────────────────
    def _clean(self, pages: list[ConfluencePage]) -> list[CleanedConfluencePage]:
        return [self._clean_page(page) for page in pages]

    def _clean_page(self, page: ConfluencePage) -> CleanedConfluencePage:
        # 라벨 → 버전 계보 메타 파생 (페이지당 1회). 라벨 없는 페이지는 전부 빈 값 → 중립 동작.
        site = parse_site(page.labels)
        product_version = parse_product_version(page.labels)
        return CleanedConfluencePage(
            page_id=page.page_id,
            title=page.title,
            space_key=page.space_key,
            markdown=self.cleaner.clean(page.html),
            html=page.html,
            last_modified=page.last_modified,
            version=page.version,
            url=page.url,
            site=site,
            product_version=product_version,
            doc_key=make_doc_key(site, page.title),
            is_eol=is_eol(site, product_version, self.settings.eol_versions),
        )

    def _chunk(self, cleaned_pages: list[CleanedConfluencePage]) -> list[ChunkedConfluencePage]:
        return [self._chunk_cleaned_page(self._mask_page(page)) for page in cleaned_pages]

    async def _prepare(self, cleaned_pages: list[CleanedConfluencePage]) -> list[ChunkedConfluencePage]:
        prepared_pages: list[ChunkedConfluencePage] = []
        for page in cleaned_pages:
            masked_page = self._mask_page(page)
            chunking_profile = self.profile_classifier.classify_page(masked_page.title, masked_page.markdown)
            chunked_page = self._chunk_cleaned_page(masked_page, chunking_profile=chunking_profile)

            # 도메인 상속 맵 — LLM 옵션이 켜지면 문서 1회 LLM 분류로 대체, 실패 시 룰 fallback.
            llm_result = await self._classify_domain(masked_page)
            if llm_result is not None:
                parent_domains = {parent.parent_id: llm_result.domain for parent in chunked_page.parents}
            else:
                # child가 자기 추론이 아니라 소속 parent의 domain을 상속하도록 parent domain 맵 전달
                parent_domains = {parent.parent_id: parent.domain for parent in chunked_page.parents}

            children = self.metadata_classifier.classify_batch(chunked_page.children, chunking_profile, parent_domains)
            if llm_result is not None:
                children = [self._apply_llm_domain_meta(child, llm_result) for child in children]

            prepared_pages.append(
                ChunkedConfluencePage(page=chunked_page.page, parents=chunked_page.parents, children=children)
            )

        return prepared_pages

    async def _classify_domain(self, page: CleanedConfluencePage) -> DomainResult | None:
        """LLM 옵션이 켜져 있으면 문서 단위 도메인을 분류한다. off·실패 시 None(룰 fallback)."""
        if not self.settings.llm_classify_enabled:
            return None
        return await self.domain_classifier.classify(page.title, page.markdown)

    def _apply_llm_domain_meta(self, child: ChildChunk, result: DomainResult) -> ChildChunk:
        """LLM 분류 출처·신뢰도를 청크에 기록하고, secondary는 ``domain2:{x}`` 보조 태그로 보존."""
        tags = list(child.tags or [])
        if result.secondary:
            tags = [f"domain2:{result.secondary}", *tags]
        return replace(child, domain_source="llm", domain_confidence=result.confidence, tags=tags)

    def _mask_page(self, page: CleanedConfluencePage) -> CleanedConfluencePage:
        return replace(page, markdown=self.masker.mask(page.markdown))

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
            site=page.site,
            product_version=page.product_version,
            doc_key=page.doc_key,
            is_eol=page.is_eol,
        )
        chunker = self.control_chunker if chunking_profile == "control_like" else self.chunker
        parents, children = chunker.chunk(markdown_page)
        return ChunkedConfluencePage(page=page, parents=parents, children=children)
