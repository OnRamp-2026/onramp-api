from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, cast

import structlog
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import EventOutbox, utcnow
from app.queue.events import StreamEnvelope, encode_envelope

logger = structlog.get_logger(__name__)


class OutboxPublisher:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        redis: Redis,
        *,
        batch_size: int = 100,
    ) -> None:
        self.session_factory = session_factory
        self.redis = redis
        self.batch_size = batch_size

    async def publish_once(self, now: datetime | None = None) -> int:
        current_time = now or utcnow()
        published = 0
        async with self.session_factory() as session:
            rows = list(
                await session.scalars(
                    select(EventOutbox)
                    .where(
                        EventOutbox.published_at.is_(None),
                        EventOutbox.available_at <= current_time,
                    )
                    .order_by(EventOutbox.created_at)
                    .limit(self.batch_size)
                    .with_for_update(skip_locked=True)
                )
            )
            for row in rows:
                envelope = StreamEnvelope(
                    event_id=row.id,
                    event_type=row.event_type,
                    payload=row.payload_json,
                )
                try:
                    await self.redis.xadd(
                        row.stream_name,
                        cast(dict[Any, Any], encode_envelope(envelope)),
                    )
                except Exception as exc:
                    row.publish_attempts += 1
                    row.last_error = type(exc).__name__
                    row.available_at = current_time + timedelta(seconds=min(60, 2**row.publish_attempts))
                    await logger.awarning(
                        "outbox_publish_failed",
                        event_id=row.id,
                        stream=row.stream_name,
                    )
                    continue
                row.published_at = current_time
                row.last_error = None
                published += 1
            await session.commit()
        return published
