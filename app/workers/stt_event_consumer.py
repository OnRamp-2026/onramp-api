from __future__ import annotations

import asyncio
import os
import socket

import structlog

from app.config import get_settings
from app.db.postgres import get_session_factory
from app.db.redis import get_redis
from app.queue.constants import (
    REPORT_EVENT_GROUP,
    STT_COMPLETED_STREAM,
    STT_PROGRESS_STREAM,
    STT_TRANSCRIPT_COMPLETED_STREAM,
    TRANSCRIPT_OBSERVER_GROUP,
    WORKFLOW_UPDATER_GROUP,
)
from app.queue.consumer import ensure_consumer_group, read_new_or_reclaimed
from app.queue.events import decode_envelope
from app.services.stt_event_service import SttEventService

logger = structlog.get_logger(__name__)


async def consume_stream(stream: str, group: str) -> None:
    settings = get_settings()
    redis = get_redis()
    service = SttEventService(get_session_factory())
    consumer = f"{socket.gethostname()}-{os.getpid()}-{group}"
    await ensure_consumer_group(redis, stream, group)
    while True:
        messages = await read_new_or_reclaimed(
            redis,
            stream=stream,
            group=group,
            consumer=consumer,
            block_ms=settings.redis_stream_block_ms,
            count=settings.redis_stream_read_count,
            reclaim_idle_ms=settings.redis_stream_reclaim_idle_ms,
        )
        for message_id, fields in messages:
            try:
                await service.process(decode_envelope(fields))
            except Exception:
                await logger.aexception("stt_event_processing_failed", stream=stream, message_id=message_id)
                continue
            await redis.xack(stream, group, message_id)


async def run() -> None:
    await asyncio.gather(
        consume_stream(STT_PROGRESS_STREAM, WORKFLOW_UPDATER_GROUP),
        consume_stream(STT_TRANSCRIPT_COMPLETED_STREAM, TRANSCRIPT_OBSERVER_GROUP),
        consume_stream(STT_COMPLETED_STREAM, REPORT_EVENT_GROUP),
    )


if __name__ == "__main__":
    asyncio.run(run())
