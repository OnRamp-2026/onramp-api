from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models import EventOutbox
from app.queue.outbox import OutboxPublisher


class FakeRedis:
    def __init__(self) -> None:
        self.messages: list[tuple[str, dict[str, Any]]] = []

    async def xadd(self, stream: str, fields: dict[str, Any]) -> str:
        self.messages.append((stream, fields))
        return "1-0"


@pytest.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_outbox_publisher_marks_event_after_xadd(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    event = EventOutbox(
        id="evt_test",
        aggregate_type="transcription",
        aggregate_id="transcription-id",
        event_type="transcription.requested",
        stream_name="onramp:stt:requests:v1",
        payload_json={"tenant_id": "tenant-a"},
    )
    async with session_factory() as session:
        session.add(event)
        await session.commit()

    redis = FakeRedis()
    publisher = OutboxPublisher(session_factory, redis)  # type: ignore[arg-type]
    published = await publisher.publish_once(now=datetime.now(UTC))

    async with session_factory() as session:
        persisted = await session.scalar(select(EventOutbox).where(EventOutbox.id == "evt_test"))

    assert published == 1
    assert redis.messages[0][0] == "onramp:stt:requests:v1"
    assert redis.messages[0][1]["event_id"] == "evt_test"
    assert persisted is not None
    assert persisted.published_at is not None
