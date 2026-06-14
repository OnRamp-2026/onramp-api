from __future__ import annotations

from typing import Any

from redis.asyncio import Redis
from redis.exceptions import ResponseError


async def ensure_consumer_group(redis: Redis, stream: str, group: str) -> None:
    try:
        await redis.xgroup_create(stream, group, id="0", mkstream=True)
    except ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


async def read_new_or_reclaimed(
    redis: Redis,
    *,
    stream: str,
    group: str,
    consumer: str,
    block_ms: int,
    count: int,
    reclaim_idle_ms: int,
) -> list[tuple[str, dict[str, str]]]:
    reclaimed = await redis.xautoclaim(
        stream,
        group,
        consumer,
        reclaim_idle_ms,
        "0-0",
        count=count,
    )
    reclaimed_messages = _messages(reclaimed[1] if len(reclaimed) > 1 else [])
    if reclaimed_messages:
        return reclaimed_messages
    response = await redis.xreadgroup(
        group,
        consumer,
        streams={stream: ">"},
        count=count,
        block=block_ms,
    )
    if not response:
        return []
    return _messages(response[0][1])


def _messages(messages: list[Any]) -> list[tuple[str, dict[str, str]]]:
    return [(str(message_id), dict(fields)) for message_id, fields in messages]
