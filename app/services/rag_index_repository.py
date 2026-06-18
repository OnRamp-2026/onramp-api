"""PostgreSQL read/write for RAG indexing (index_run, source_document, chunk_registry)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from app.db.models import (
    ChunkRegistry,
    IndexRun,
    IndexRunStage,
    IndexRunStatus,
    IndexRunTrigger,
    IndexRunType,
    SourceDocument,
    SourceDocumentPrevious,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.rag.chunker import ChildChunk, ParentChunk
    from app.services.ingest_service import CleanedConfluencePage


def _now() -> datetime:
    return datetime.now(UTC)


# ── index_run ──────────────────────────────────────────────────────────────────


async def create_index_run(
    db: AsyncSession,
    *,
    tenant_id: str,
    run_type: str = IndexRunType.incremental.value,
    trigger: str = IndexRunTrigger.manual.value,
) -> IndexRun:
    run = IndexRun(
        tenant_id=tenant_id,
        run_type=run_type,
        trigger=trigger,
        status=IndexRunStatus.running.value,
        stage=IndexRunStage.indexing.value,
    )
    db.add(run)
    await db.flush()
    return run


async def enqueue_index_run(
    db: AsyncSession,
    *,
    tenant_id: str,
    run_type: str,
    trigger: str,
) -> IndexRun | None:
    if await get_active_index_run(db, tenant_id=tenant_id) is not None:
        return None
    run = IndexRun(
        tenant_id=tenant_id,
        run_type=run_type,
        trigger=trigger,
        status=IndexRunStatus.queued.value,
        stage=IndexRunStage.queued.value,
    )
    db.add(run)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        return None
    await db.refresh(run)
    return run


async def claim_next_index_run(db: AsyncSession) -> IndexRun | None:
    run = await db.scalar(
        select(IndexRun)
        .where(IndexRun.status == IndexRunStatus.queued.value)
        .order_by(IndexRun.created_at)
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    if run is None:
        return None
    run.status = IndexRunStatus.running.value
    run.stage = IndexRunStage.fetching.value
    run.started_at = _now()
    await db.commit()
    await db.refresh(run)
    return run


async def get_active_index_run(db: AsyncSession, *, tenant_id: str) -> IndexRun | None:
    return cast(
        IndexRun | None,
        await db.scalar(
            select(IndexRun)
            .where(
                IndexRun.tenant_id == tenant_id,
                IndexRun.status.in_((IndexRunStatus.queued.value, IndexRunStatus.running.value)),
            )
            .order_by(IndexRun.created_at.desc())
            .limit(1)
        ),
    )


async def list_index_runs(db: AsyncSession, *, tenant_id: str, limit: int = 10) -> list[IndexRun]:
    rows = await db.scalars(
        select(IndexRun).where(IndexRun.tenant_id == tenant_id).order_by(IndexRun.created_at.desc()).limit(limit)
    )
    return list(rows)


async def update_index_run_progress(
    db: AsyncSession,
    run: IndexRun,
    *,
    stage: str | None = None,
    pages_discovered: int | None = None,
    pages_processed: int | None = None,
    pages_indexed: int | None = None,
    pages_skipped: int | None = None,
    pages_failed: int | None = None,
    chunks_indexed: int | None = None,
    chunks_deleted: int | None = None,
) -> None:
    values = {
        "stage": stage,
        "pages_discovered": pages_discovered,
        "pages_processed": pages_processed,
        "pages_indexed": pages_indexed,
        "pages_skipped": pages_skipped,
        "pages_failed": pages_failed,
        "chunks_indexed": chunks_indexed,
        "chunks_deleted": chunks_deleted,
    }
    for name, value in values.items():
        if value is not None:
            setattr(run, name, value)
    await db.commit()


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
    run.stage = IndexRunStage.completed.value
    run.finished_at = _now()
    run.pages_indexed = pages_indexed
    run.pages_processed = run.pages_skipped + pages_indexed + pages_failed
    run.pages_failed = pages_failed
    run.chunks_indexed = chunks_indexed
    run.chunks_deleted = chunks_deleted
    await db.commit()


async def fail_index_run(db: AsyncSession, run: IndexRun, *, error: str) -> None:
    run.status = IndexRunStatus.failed.value
    run.stage = IndexRunStage.completed.value
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
    source: str = "confluence",
) -> bool:
    """True = hash changed or new page → indexing needed."""
    stored_hash = await db.scalar(
        select(SourceDocument.cleaned_markdown_hash).where(
            SourceDocument.tenant_id == tenant_id,
            SourceDocument.source == source,
            SourceDocument.page_id == page_id,
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
    source: str = "confluence",
) -> None:
    """Rotate current → previous, then upsert new current. All within one flush."""
    now = _now()
    existing = await db.scalar(
        select(SourceDocument).where(
            SourceDocument.tenant_id == tenant_id,
            SourceDocument.source == source,
            SourceDocument.page_id == page.page_id,
        )
    )

    if existing is not None:
        # delete old previous (if any), then promote current → previous
        await db.execute(
            delete(SourceDocumentPrevious).where(
                SourceDocumentPrevious.tenant_id == tenant_id,
                SourceDocumentPrevious.source == source,
                SourceDocumentPrevious.page_id == page.page_id,
            )
        )
        db.add(
            SourceDocumentPrevious(
                tenant_id=existing.tenant_id,
                page_id=existing.page_id,
                source=existing.source,
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
            SourceDocument(
                tenant_id=tenant_id,
                source=source,
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
    source: str = "confluence",
    parents: list[ParentChunk] | None = None,
) -> None:
    if not children:
        return
    if parents is None:
        # parents 미전달 시 merge가 parent_content=None으로 기존 값을 조용히 덮어쓴다 → fail-fast.
        raise ValueError("upsert_chunk_registry: parents must be provided (None이면 parent_content가 소거됨)")
    from app.rag.indexer import _point_id

    # parent expansion(#212 Phase 0-A): child row에 소속 parent 본문을 채운다(조회 시 parent_id로 dedupe).
    parent_content = {p.parent_id: p.content for p in parents}
    for c in children:
        await db.merge(
            ChunkRegistry(
                chunk_id=c.chunk_id,
                tenant_id=tenant_id,
                point_id=uuid.UUID(_point_id(c.chunk_id)),
                parent_id=c.parent_id,
                page_id=c.page_id,
                source=source,
                run_id=run_id,
                domain=c.domain,
                token_count=c.token_count,
                hash=c.hash,
                parent_content=parent_content.get(c.parent_id),
            )
        )
    await db.flush()


async def get_parent_contexts(db: AsyncSession, *, tenant_id: str, parent_ids: list[str]) -> dict[str, str]:
    """parent_id → parent_content 조회 (#212 Phase 0-A).

    chunk_registry는 child별 row라 같은 parent가 여러 행에 중복된다 → parent_id로 dedupe.
    저장 위치를 호출부에서 감추는 wrapper — Phase 0-B에서 전용 parent 테이블로 바꿔도 시그니처는 유지한다.
    """
    if not parent_ids:
        return {}
    rows = await db.execute(
        select(ChunkRegistry.parent_id, ChunkRegistry.parent_content).where(
            ChunkRegistry.tenant_id == tenant_id,
            ChunkRegistry.parent_id.in_(set(parent_ids)),
            ChunkRegistry.parent_content.isnot(None),
        )
    )
    out: dict[str, str] = {}
    for parent_id, content in rows.all():
        if content and parent_id not in out:
            out[parent_id] = content
    return out


async def collect_stale_chunks(
    db: AsyncSession,
    *,
    tenant_id: str,
    page_id: str,
    run_id: uuid.UUID,
    source: str = "confluence",
) -> list[tuple[str, uuid.UUID]]:
    """Returns (chunk_id, point_id) for chunks that belong to a previous run."""
    rows = await db.execute(
        select(ChunkRegistry.chunk_id, ChunkRegistry.point_id).where(
            ChunkRegistry.tenant_id == tenant_id,
            ChunkRegistry.source == source,
            ChunkRegistry.page_id == page_id,
            ChunkRegistry.run_id != run_id,
        )
    )
    return [(chunk_id, point_id) for chunk_id, point_id in rows.all()]


async def delete_stale_chunk_rows(db: AsyncSession, *, chunk_ids: list[str]) -> None:
    if not chunk_ids:
        return
    await db.execute(delete(ChunkRegistry).where(ChunkRegistry.chunk_id.in_(chunk_ids)))
    await db.flush()
