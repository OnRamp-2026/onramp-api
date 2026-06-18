from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models import IndexRunStatus
from app.services.ingestion_run_service import enqueue_run, normalize_run_type


@pytest.fixture
async def db() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


def test_normalize_run_type() -> None:
    assert normalize_run_type("incremental") == "incremental"
    assert normalize_run_type("full_scan") == "full_scan"
    with pytest.raises(ValueError):
        normalize_run_type("force")


async def test_enqueue_run_is_tenant_scoped_and_deduplicated(db: AsyncSession) -> None:
    first = await enqueue_run(db, tenant_id="tenant-a", mode="incremental")
    duplicate = await enqueue_run(db, tenant_id="tenant-a", mode="full_scan")
    other_tenant = await enqueue_run(db, tenant_id="tenant-b", mode="full_scan")

    assert first is not None
    assert first.status == IndexRunStatus.queued.value
    assert duplicate is None
    assert other_tenant is not None
