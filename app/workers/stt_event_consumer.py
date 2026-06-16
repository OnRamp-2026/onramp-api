from __future__ import annotations

import asyncio
import os
import socket
from json import JSONDecodeError

import structlog
from pydantic import ValidationError
from redis.asyncio import Redis

from app.config import get_settings
from app.db.postgres import get_session_factory
from app.db.redis import get_stt_redis
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
from app.services.stt_event_service import SttEventService, UnrecoverableSttEventError

logger = structlog.get_logger(__name__)


def consumer_group_name(base_group: str, suffix: str) -> str:
    normalized = suffix.strip()
    return f"{base_group}-{normalized}" if normalized else base_group


async def process_message(
    redis: Redis,
    service: SttEventService,
    *,
    stream: str,
    group: str,
    message_id: str,
    fields: dict[str, str],
) -> None:
    try:
        await service.process(decode_envelope(fields))
    except (JSONDecodeError, KeyError, ValidationError, UnrecoverableSttEventError):
        await logger.aexception(
            "stt_event_processing_unrecoverable",
            stream=stream,
            message_id=message_id,
        )
        await redis.xack(stream, group, message_id)
        return
    except Exception:
        await logger.aexception("stt_event_processing_failed", stream=stream, message_id=message_id)
        return
    await redis.xack(stream, group, message_id)


async def consume_stream(stream: str, group: str) -> None:
    settings = get_settings()
    redis = get_stt_redis()
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
            await process_message(
                redis,
                service,
                stream=stream,
                group=group,
                message_id=message_id,
                fields=fields,
            )


async def run() -> None:
    settings = get_settings()
    await asyncio.gather(
        consume_stream(
            STT_PROGRESS_STREAM,
            consumer_group_name(WORKFLOW_UPDATER_GROUP, settings.stt_consumer_group_suffix),
        ),
        consume_stream(
            STT_TRANSCRIPT_COMPLETED_STREAM,
            consumer_group_name(TRANSCRIPT_OBSERVER_GROUP, settings.stt_consumer_group_suffix),
        ),
        consume_stream(
            STT_COMPLETED_STREAM,
            consumer_group_name(REPORT_EVENT_GROUP, settings.stt_consumer_group_suffix),
        ),
    )


if __name__ == "__main__":
    asyncio.run(run())
