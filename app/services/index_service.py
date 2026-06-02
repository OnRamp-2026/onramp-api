"""Index prepared Confluence chunks into Qdrant."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from qdrant_client import QdrantClient

from app.config import Settings
from app.rag.chunker import ChildChunk
from app.rag.embedder import Embedder
from app.rag.indexer import index_children
from app.services.ingest_service import ChunkedConfluencePage, IngestService

IndexChildrenFn = Callable[
    [list[ChildChunk]],
    Awaitable[int],
]


@dataclass(frozen=True)
class IndexResult:
    """Summary of one recent Confluence indexing run."""

    pages_indexed: int
    chunks_indexed: int


class IndexService:
    """Prepare recent Confluence pages and upsert their child chunks."""

    def __init__(
        self,
        ingest_service: IngestService | None = None,
        embedder: Embedder | None = None,
        client: QdrantClient | None = None,
        settings: Settings | None = None,
        index_children_fn: IndexChildrenFn | None = None,
    ) -> None:
        self.ingest_service = ingest_service or IngestService()
        self.embedder = embedder
        self.client = client
        self.settings = settings
        self.index_children_fn = index_children_fn

    async def index_recent_pages(self, hours: int = 24, limit: int = 50) -> IndexResult:
        """Prepare recent pages and upsert all child chunks into Qdrant."""

        pages = await self.ingest_service.prepare_recent_pages_for_embedding(hours=hours, limit=limit)
        children = self._flatten_children(pages)
        chunks_indexed = await self._index_children(children)
        return IndexResult(pages_indexed=len(pages), chunks_indexed=chunks_indexed)

    def _flatten_children(self, pages: list[ChunkedConfluencePage]) -> list[ChildChunk]:
        return [child for page in pages for child in page.children]

    async def _index_children(self, children: list[ChildChunk]) -> int:
        if self.index_children_fn is not None:
            return await self.index_children_fn(children)
        return await index_children(
            children,
            embedder=self.embedder,
            client=self.client,
            settings=self.settings,
        )
