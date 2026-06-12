from __future__ import annotations

import asyncio

import structlog

from app.config import get_settings
from app.db.postgres import get_session_factory
from app.db.redis import get_redis
from app.queue.outbox import OutboxPublisher

logger = structlog.get_logger(__name__)


async def run() -> None:
    settings = get_settings()
    publisher = OutboxPublisher(
        get_session_factory(),
        get_redis(),
        batch_size=settings.redis_outbox_batch_size,
    )
    poll_interval = settings.redis_outbox_poll_interval_ms / 1000
    while True:
        try:
            published = await publisher.publish_once()
        except Exception:
            await logger.aexception("outbox_publish_loop_failed")
            await asyncio.sleep(poll_interval)
            continue
        if published == 0:
            await asyncio.sleep(poll_interval)


if __name__ == "__main__":
    asyncio.run(run())
