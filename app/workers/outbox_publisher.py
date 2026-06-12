from __future__ import annotations

import asyncio

from app.config import get_settings
from app.db.postgres import get_session_factory
from app.db.redis import get_redis
from app.queue.outbox import OutboxPublisher


async def run() -> None:
    settings = get_settings()
    publisher = OutboxPublisher(
        get_session_factory(),
        get_redis(),
        batch_size=settings.redis_outbox_batch_size,
    )
    poll_interval = settings.redis_outbox_poll_interval_ms / 1000
    while True:
        published = await publisher.publish_once()
        if published == 0:
            await asyncio.sleep(poll_interval)


if __name__ == "__main__":
    asyncio.run(run())
