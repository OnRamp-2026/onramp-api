from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.models.transcription import TranscriptionCreateRequest
from app.services.transcription_service import create_workflow
from tests.unit.test_transcription_workflow import FakeSttResultClient


@pytest.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_same_idempotency_key_isolated_by_user(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    request = TranscriptionCreateRequest(
        filename="meeting.m4a",
        content_type="audio/mp4",
        size_bytes=1024,
        title="회의",
        language="ko-KR",
        category="장애대응",
    )
    client = FakeSttResultClient()

    async with session_factory() as session:
        first, _ = await create_workflow(
            session,
            client,
            tenant_id="tenant-a",
            user_id="user-a",
            idempotency_key="same-key",
            request=request,
        )
        await session.commit()

    async with session_factory() as session:
        second, created = await create_workflow(
            session,
            client,
            tenant_id="tenant-a",
            user_id="user-b",
            idempotency_key="same-key",
            request=request,
        )

    assert isinstance(first.workflow.transcription_id, UUID)
    assert created is True
    assert second.workflow.id != first.workflow.id
