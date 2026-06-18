"""Index prepared Confluence chunks into Qdrant, OpenSearch, and PostgreSQL."""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial

import anyio
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

from app.config import Settings, get_settings
from app.db.models import IndexRun
from app.db.postgres import session_scope as _default_session_scope
from app.db.qdrant import get_qdrant
from app.rag.chunker import ChildChunk
from app.rag.embedder import Embedder
from app.rag.indexer import index_children
from app.services import rag_index_repository as repo
from app.services.ingest_service import ChunkedConfluencePage, IngestService

logger = logging.getLogger(__name__)

IndexChildrenFn = Callable[
    [list[ChildChunk]],
    Awaitable[int],
]
ProgressFn = Callable[[dict[str, int | str]], Awaitable[None]]


@dataclass(frozen=True)
class IndexResult:
    """Summary of one Confluence indexing run."""

    pages_indexed: int
    chunks_indexed: int
    chunks_deleted: int = field(default=0)
    pages_skipped: int = field(default=0)


class IndexService:
    """Fetch, embed, and upsert Confluence pages into Qdrant/OpenSearch/PostgreSQL."""

    def __init__(
        self,
        ingest_service: IngestService | None = None,
        embedder: Embedder | None = None,
        client: QdrantClient | None = None,
        settings: Settings | None = None,
        index_children_fn: IndexChildrenFn | None = None,
        session_factory=None,
    ) -> None:
        self.ingest_service = ingest_service or IngestService()
        self.embedder = embedder
        self.client = client
        self.settings = settings
        self.index_children_fn = index_children_fn
        self._session_factory = session_factory or _default_session_scope

    async def index_recent_pages(
        self,
        hours: int = 24,
        limit: int = 50,
        *,
        force: bool = False,
        run_id: uuid.UUID | None = None,
        progress: ProgressFn | None = None,
    ) -> IndexResult:
        pages = await self.ingest_service.prepare_recent_pages_for_embedding(hours=hours, limit=limit)
        return await self.index_prepared(pages, force=force, run_id=run_id, progress=progress)

    async def index_all_pages(
        self,
        limit: int = 50,
        *,
        force: bool = False,
        run_id: uuid.UUID | None = None,
        progress: ProgressFn | None = None,
    ) -> IndexResult:
        pages = await self.ingest_service.prepare_all_pages_for_embedding(limit=limit)
        return await self.index_prepared(pages, force=force, run_id=run_id, progress=progress)

    async def index_prepared(
        self,
        pages: list[ChunkedConfluencePage],
        *,
        source: str = "confluence",
        force: bool = False,
        run_id: uuid.UUID | None = None,
        progress: ProgressFn | None = None,
    ) -> IndexResult:
        """Index already-prepared chunks. ``source`` 는 멀티소스 원장 식별키 (confluence|github).

        ``force=True`` 면 content-hash dedup(should_index_page)을 건너뛰고 모든 페이지를 재색인한다
        (도메인 분류 방식만 바꿔 다시 분류·임베딩할 때 사용 — 전체 wipe 없이 재색인).
        """
        settings = self.settings or get_settings()
        tenant_id = settings.auth_default_tenant

        pages_indexed = 0
        pages_skipped = 0
        pages_failed = 0
        chunks_indexed = 0
        chunks_deleted = 0

        async with self._session_factory() as db:
            run = await db.get(IndexRun, run_id) if run_id is not None else None
            if run is None:
                run = await repo.create_index_run(db, tenant_id=tenant_id)
            if progress is not None:
                await progress({"stage": "indexing", "pages_discovered": len(pages)})
            try:
                for chunked_page in pages:
                    pg = chunked_page.page
                    try:
                        raw_html_hash = hashlib.sha256(pg.html.encode()).hexdigest()
                        cleaned_markdown_hash = hashlib.sha256(pg.markdown.encode()).hexdigest()

                        if not force and not await repo.should_index_page(
                            db,
                            tenant_id=tenant_id,
                            page_id=pg.page_id,
                            cleaned_markdown_hash=cleaned_markdown_hash,
                            source=source,
                        ):
                            pages_skipped += 1
                            if progress is not None:
                                await progress(
                                    {
                                        "stage": "indexing",
                                        "pages_discovered": len(pages),
                                        "pages_processed": pages_indexed + pages_skipped + pages_failed,
                                        "pages_indexed": pages_indexed,
                                        "pages_skipped": pages_skipped,
                                        "pages_failed": pages_failed,
                                        "chunks_indexed": chunks_indexed,
                                        "chunks_deleted": chunks_deleted,
                                    }
                                )
                            continue

                        # 1. Qdrant / OpenSearch upsert (search index first — idempotent)
                        n = await self._index_children(chunked_page.children)

                        # 2. rotate snapshot + save document in PostgreSQL
                        await repo.rotate_and_save_document(
                            db,
                            tenant_id=tenant_id,
                            page=pg,
                            raw_html_hash=raw_html_hash,
                            cleaned_markdown_hash=cleaned_markdown_hash,
                            chunk_count=len(chunked_page.children),
                            source=source,
                        )

                        # 3. upsert chunk_registry
                        await repo.upsert_chunk_registry(
                            db,
                            children=chunked_page.children,
                            run_id=run.run_id,
                            tenant_id=tenant_id,
                            source=source,
                            parents=chunked_page.parents,
                        )

                        # 4. delete stale chunks from search + registry
                        stale = await repo.collect_stale_chunks(
                            db,
                            tenant_id=tenant_id,
                            page_id=pg.page_id,
                            run_id=run.run_id,
                            source=source,
                        )
                        if stale:
                            stale_chunk_ids = [cid for cid, _ in stale]
                            stale_point_ids = [pid for _, pid in stale]
                            await self._delete_stale_from_search(stale_chunk_ids, stale_point_ids, settings)
                            await repo.delete_stale_chunk_rows(db, chunk_ids=stale_chunk_ids)
                            chunks_deleted += len(stale)

                        chunks_indexed += n
                        pages_indexed += 1
                        if progress is not None:
                            await progress(
                                {
                                    "stage": "indexing",
                                    "pages_discovered": len(pages),
                                    "pages_processed": pages_indexed + pages_skipped + pages_failed,
                                    "pages_indexed": pages_indexed,
                                    "pages_skipped": pages_skipped,
                                    "pages_failed": pages_failed,
                                    "chunks_indexed": chunks_indexed,
                                    "chunks_deleted": chunks_deleted,
                                }
                            )

                    except Exception:
                        logger.exception("Failed to index page %s", pg.page_id)
                        pages_failed += 1
                        if progress is not None:
                            await progress(
                                {
                                    "stage": "indexing",
                                    "pages_discovered": len(pages),
                                    "pages_processed": pages_indexed + pages_skipped + pages_failed,
                                    "pages_indexed": pages_indexed,
                                    "pages_skipped": pages_skipped,
                                    "pages_failed": pages_failed,
                                    "chunks_indexed": chunks_indexed,
                                    "chunks_deleted": chunks_deleted,
                                }
                            )

                run.pages_skipped = pages_skipped
                run.pages_discovered = len(pages)
                await repo.finish_index_run(
                    db,
                    run,
                    pages_indexed=pages_indexed,
                    pages_failed=pages_failed,
                    chunks_indexed=chunks_indexed,
                    chunks_deleted=chunks_deleted,
                )
            except Exception:
                logger.exception("Index run failed unexpectedly")
                await repo.fail_index_run(db, run, error="Unexpected error in _index_prepared")
                raise

        return IndexResult(
            pages_indexed=pages_indexed,
            chunks_indexed=chunks_indexed,
            chunks_deleted=chunks_deleted,
            pages_skipped=pages_skipped,
        )

    async def _delete_stale_from_search(
        self,
        chunk_ids: list[str],
        point_ids: list[uuid.UUID],
        settings: Settings,
    ) -> None:
        client = self.client or get_qdrant()
        try:
            await anyio.to_thread.run_sync(
                partial(
                    client.delete,
                    collection_name=settings.qdrant_collection,
                    points_selector=PointIdsList(points=[str(pid) for pid in point_ids]),
                )
            )
        except Exception:
            logger.exception("Qdrant stale chunk delete failed (non-fatal)")

        if settings.bm25_search_enabled:
            from app.db.opensearch import get_opensearch

            try:
                await get_opensearch().delete_chunks(chunk_ids)
            except Exception:
                logger.exception("OpenSearch stale chunk delete failed (non-fatal)")

    async def _index_children(self, children: list[ChildChunk]) -> int:
        if self.index_children_fn is not None:
            return await self.index_children_fn(children)
        return await index_children(
            children,
            embedder=self.embedder,
            client=self.client,
            settings=self.settings,
        )
