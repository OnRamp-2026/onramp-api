"""Unit tests for rag_index_repository — in-memory SQLite via aiosqlite."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models import ChunkRegistry, ConfluenceDocument, ConfluenceDocumentPrevious, IndexRunStatus
from app.services import rag_index_repository as repo

# ── in-memory DB fixture ───────────────────────────────────────────────────────

@pytest.fixture
async def db() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


# ── fake ChildChunk ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Chunk:
    chunk_id: str
    parent_id: str = "p"
    page_id: str = "pg1"
    domain: str | None = "장애대응"
    token_count: int = 10
    hash: str = "abc"
    embedding_text: str = "test"
    content: str = "내용"
    content_vector: list[float] = field(default_factory=list)
    block_types: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    code_languages: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Page:
    page_id: str = "pg1"
    space_key: str = "ONR"
    title: str = "테스트 페이지"
    url: str = "http://x"
    html: str = "<p>hello</p>"
    markdown: str = "hello"
    version: int | None = 1
    last_modified: str = "2026-06-16T00:00:00Z"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ── index_run ──────────────────────────────────────────────────────────────────

async def test_create_and_finish_index_run(db: AsyncSession) -> None:
    run = await repo.create_index_run(db, tenant_id="onramp")
    assert run.run_id is not None
    assert run.status == IndexRunStatus.running.value

    await repo.finish_index_run(db, run, pages_indexed=3, pages_failed=0, chunks_indexed=10, chunks_deleted=2)
    assert run.status == IndexRunStatus.success.value
    assert run.pages_indexed == 3
    assert run.chunks_deleted == 2


async def test_fail_index_run(db: AsyncSession) -> None:
    run = await repo.create_index_run(db, tenant_id="onramp")
    await repo.fail_index_run(db, run, error="boom")
    assert run.status == IndexRunStatus.failed.value
    assert run.error_message == "boom"


# ── should_index_page ──────────────────────────────────────────────────────────

async def test_should_index_page_new(db: AsyncSession) -> None:
    assert await repo.should_index_page(db, tenant_id="onramp", page_id="pg1", cleaned_markdown_hash="aaa") is True


async def test_should_index_page_same_hash(db: AsyncSession) -> None:
    page = _Page()
    md_hash = _hash(page.markdown)
    raw_hash = _hash(page.html)
    await repo.rotate_and_save_document(
        db, tenant_id="onramp", page=page,
        raw_html_hash=raw_hash, cleaned_markdown_hash=md_hash, chunk_count=1,
    )
    assert await repo.should_index_page(db, tenant_id="onramp", page_id="pg1", cleaned_markdown_hash=md_hash) is False


async def test_should_index_page_changed_hash(db: AsyncSession) -> None:
    page = _Page()
    md_hash = _hash(page.markdown)
    raw_hash = _hash(page.html)
    await repo.rotate_and_save_document(
        db, tenant_id="onramp", page=page,
        raw_html_hash=raw_hash, cleaned_markdown_hash=md_hash, chunk_count=1,
    )
    assert await repo.should_index_page(db, tenant_id="onramp", page_id="pg1", cleaned_markdown_hash="changed") is True


# ── rotate_and_save_document ───────────────────────────────────────────────────

async def test_rotate_creates_new_document(db: AsyncSession) -> None:
    page = _Page()
    await repo.rotate_and_save_document(
        db, tenant_id="onramp", page=page,
        raw_html_hash=_hash(page.html), cleaned_markdown_hash=_hash(page.markdown), chunk_count=2,
    )
    doc = await db.get(ConfluenceDocument, ("onramp", "pg1"))
    assert doc is not None
    assert doc.raw_html == page.html
    assert doc.chunk_count == 2


async def test_rotate_promotes_current_to_previous(db: AsyncSession) -> None:
    page_v1 = _Page(html="<p>v1</p>", markdown="v1")
    await repo.rotate_and_save_document(
        db, tenant_id="onramp", page=page_v1,
        raw_html_hash=_hash(page_v1.html), cleaned_markdown_hash=_hash(page_v1.markdown), chunk_count=1,
    )

    page_v2 = _Page(html="<p>v2</p>", markdown="v2")
    await repo.rotate_and_save_document(
        db, tenant_id="onramp", page=page_v2,
        raw_html_hash=_hash(page_v2.html), cleaned_markdown_hash=_hash(page_v2.markdown), chunk_count=2,
    )

    current = await db.get(ConfluenceDocument, ("onramp", "pg1"))
    previous = await db.get(ConfluenceDocumentPrevious, ("onramp", "pg1"))
    assert current.raw_html == "<p>v2</p>"
    assert previous.raw_html == "<p>v1</p>"


async def test_rotate_replaces_old_previous(db: AsyncSession) -> None:
    for i in range(1, 4):
        p = _Page(html=f"<p>v{i}</p>", markdown=f"v{i}")
        await repo.rotate_and_save_document(
            db, tenant_id="onramp", page=p,
            raw_html_hash=_hash(p.html), cleaned_markdown_hash=_hash(p.markdown), chunk_count=i,
        )

    current = await db.get(ConfluenceDocument, ("onramp", "pg1"))
    previous = await db.get(ConfluenceDocumentPrevious, ("onramp", "pg1"))
    assert current.raw_html == "<p>v3</p>"
    assert previous.raw_html == "<p>v2</p>"  # v1 is gone


# ── tenant isolation ───────────────────────────────────────────────────────────

async def test_tenant_isolation(db: AsyncSession) -> None:
    for tenant in ("t1", "t2"):
        p = _Page(markdown=f"content for {tenant}")
        await repo.rotate_and_save_document(
            db, tenant_id=tenant, page=p,
            raw_html_hash=_hash(p.html), cleaned_markdown_hash=_hash(p.markdown), chunk_count=1,
        )

    assert await repo.should_index_page(db, tenant_id="t1", page_id="pg1", cleaned_markdown_hash=_hash("content for t1")) is False
    assert await repo.should_index_page(db, tenant_id="t1", page_id="pg1", cleaned_markdown_hash=_hash("content for t2")) is True


# ── chunk_registry ─────────────────────────────────────────────────────────────

async def test_upsert_chunk_registry(db: AsyncSession) -> None:
    page = _Page()
    await repo.rotate_and_save_document(
        db, tenant_id="onramp", page=page,
        raw_html_hash=_hash(page.html), cleaned_markdown_hash=_hash(page.markdown), chunk_count=1,
    )
    run = await repo.create_index_run(db, tenant_id="onramp")
    chunks = [_Chunk(chunk_id="pg1_000"), _Chunk(chunk_id="pg1_001")]
    await repo.upsert_chunk_registry(db, children=chunks, run_id=run.run_id, tenant_id="onramp")

    row = await db.get(ChunkRegistry, "pg1_000")
    assert row is not None
    assert row.run_id == run.run_id


async def test_collect_and_delete_stale_chunks(db: AsyncSession) -> None:
    page = _Page()
    await repo.rotate_and_save_document(
        db, tenant_id="onramp", page=page,
        raw_html_hash=_hash(page.html), cleaned_markdown_hash=_hash(page.markdown), chunk_count=1,
    )
    run1 = await repo.create_index_run(db, tenant_id="onramp")
    run2 = await repo.create_index_run(db, tenant_id="onramp")

    old_chunks = [_Chunk(chunk_id="pg1_old")]
    await repo.upsert_chunk_registry(db, children=old_chunks, run_id=run1.run_id, tenant_id="onramp")

    new_chunks = [_Chunk(chunk_id="pg1_new")]
    await repo.upsert_chunk_registry(db, children=new_chunks, run_id=run2.run_id, tenant_id="onramp")

    stale = await repo.collect_stale_chunks(db, tenant_id="onramp", page_id="pg1", run_id=run2.run_id)
    assert len(stale) == 1
    assert stale[0][0] == "pg1_old"

    await repo.delete_stale_chunk_rows(db, chunk_ids=["pg1_old"])
    gone = await db.get(ChunkRegistry, "pg1_old")
    assert gone is None
