"""PostgreSQL read/write for RAG indexing (index_run, confluence_document, chunk_registry)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import delete, select

from app.db.models import (
    ChunkRegistry,
    ConfluenceDocument,
    ConfluenceDocumentPrevious,
    IndexRun,
    IndexRunStatus,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.rag.chunker import ChildChunk
    from app.services.ingest_service import CleanedConfluencePage


def _now() -> datetime:
    return datetime.now(UTC)


# ── index_run ──────────────────────────────────────────────────────────────────


async def create_index_run(db: AsyncSession, *, tenant_id: str) -> IndexRun:
    run = IndexRun(tenant_id=tenant_id)
    db.add(run)
    await db.flush()
    return run


async def finish_index_run(
    db: AsyncSession,
    run: IndexRun,
    *,
    pages_indexed: int,
    pages_failed: int,
    chunks_indexed: int,
    chunks_deleted: int,
) -> None:
    run.status = IndexRunStatus.success.value
    run.finished_at = _now()
    run.pages_indexed = pages_indexed
    run.pages_failed = pages_failed
    run.chunks_indexed = chunks_indexed
    run.chunks_deleted = chunks_deleted
    await db.commit()


async def fail_index_run(db: AsyncSession, run: IndexRun, *, error: str) -> None:
    run.status = IndexRunStatus.failed.value
    run.finished_at = _now()
    run.error_message = error[:1000]
    await db.commit()


# ── confluence_document ────────────────────────────────────────────────────────


async def should_index_page(
    db: AsyncSession,
    *,
    tenant_id: str,
    page_id: str,
    cleaned_markdown_hash: str,
) -> bool:
    """True = hash changed or new page → indexing needed."""
    stored_hash = await db.scalar(
        select(ConfluenceDocument.cleaned_markdown_hash).where(
            ConfluenceDocument.tenant_id == tenant_id,
            ConfluenceDocument.page_id == page_id,
        )
    )
    return stored_hash != cleaned_markdown_hash


async def rotate_and_save_document(
    db: AsyncSession,
    *,
    tenant_id: str,
    page: CleanedConfluencePage,
    raw_html_hash: str,
    cleaned_markdown_hash: str,
    chunk_count: int,
) -> None:
    """Rotate current → previous, then upsert new current. All within one flush."""
    now = _now()
    existing = await db.scalar(
        select(ConfluenceDocument).where(
            ConfluenceDocument.tenant_id == tenant_id,
            ConfluenceDocument.page_id == page.page_id,
        )
    )

    if existing is not None:
        # delete old previous (if any), then promote current → previous
        await db.execute(
            delete(ConfluenceDocumentPrevious).where(
                ConfluenceDocumentPrevious.tenant_id == tenant_id,
                ConfluenceDocumentPrevious.page_id == page.page_id,
            )
        )
        db.add(
            ConfluenceDocumentPrevious(
                tenant_id=existing.tenant_id,
                page_id=existing.page_id,
                space_key=existing.space_key,
                title=existing.title,
                source_url=existing.source_url,
                domain=existing.domain,
                version=existing.version,
                raw_html=existing.raw_html,
                cleaned_markdown=existing.cleaned_markdown,
                raw_html_hash=existing.raw_html_hash,
                cleaned_markdown_hash=existing.cleaned_markdown_hash,
                chunk_count=existing.chunk_count,
                last_modified=existing.last_modified,
                indexed_at=existing.indexed_at,
                replaced_at=now,
                created_at=existing.created_at,
                updated_at=now,
            )
        )
        existing.space_key = page.space_key
        existing.title = page.title
        existing.source_url = page.url
        existing.version = str(page.version) if page.version else None
        existing.raw_html = page.html
        existing.cleaned_markdown = page.markdown
        existing.raw_html_hash = raw_html_hash
        existing.cleaned_markdown_hash = cleaned_markdown_hash
        existing.chunk_count = chunk_count
        existing.indexed_at = now
        existing.updated_at = now
    else:
        db.add(
            ConfluenceDocument(
                tenant_id=tenant_id,
                page_id=page.page_id,
                space_key=page.space_key,
                title=page.title,
                source_url=page.url,
                version=str(page.version) if page.version else None,
                raw_html=page.html,
                cleaned_markdown=page.markdown,
                raw_html_hash=raw_html_hash,
                cleaned_markdown_hash=cleaned_markdown_hash,
                chunk_count=chunk_count,
                indexed_at=now,
            )
        )

    await db.flush()


# ── chunk_registry ─────────────────────────────────────────────────────────────


async def upsert_chunk_registry(
    db: AsyncSession,
    *,
    children: list[ChildChunk],
    run_id: uuid.UUID,
    tenant_id: str,
) -> None:
    if not children:
        return
    from app.rag.indexer import _point_id

    for c in children:
        await db.merge(
            ChunkRegistry(
                chunk_id=c.chunk_id,
                tenant_id=tenant_id,
                point_id=uuid.UUID(_point_id(c.chunk_id)),
                parent_id=c.parent_id,
                page_id=c.page_id,
                run_id=run_id,
                domain=c.domain,
                token_count=c.token_count,
                hash=c.hash,
            )
        )
    await db.flush()


async def collect_stale_chunks(
    db: AsyncSession,
    *,
    tenant_id: str,
    page_id: str,
    run_id: uuid.UUID,
) -> list[tuple[str, uuid.UUID]]:
    """Returns (chunk_id, point_id) for chunks that belong to a previous run."""
    rows = await db.execute(
        select(ChunkRegistry.chunk_id, ChunkRegistry.point_id).where(
            ChunkRegistry.tenant_id == tenant_id,
            ChunkRegistry.page_id == page_id,
            ChunkRegistry.run_id != run_id,
        )
    )
    return list(rows.all())


async def delete_stale_chunk_rows(db: AsyncSession, *, chunk_ids: list[str]) -> None:
    if not chunk_ids:
        return
    await db.execute(delete(ChunkRegistry).where(ChunkRegistry.chunk_id.in_(chunk_ids)))
    await db.flush()
