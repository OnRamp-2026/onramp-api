from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

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
    raw_response = cast(Any, response)
    messages: object
    if isinstance(raw_response, Mapping):
        messages = next(iter(raw_response.values()), [])
    else:
        messages = raw_response[0][1]
    return _messages(messages)


def _messages(messages: object) -> list[tuple[str, dict[str, str]]]:
    if not isinstance(messages, Sequence):
        raise ValueError("Redis stream messages must be a sequence")

    normalized: list[tuple[str, dict[str, str]]] = []
    for message in messages:
        if not isinstance(message, Sequence) or len(message) != 2:
            raise ValueError("Redis stream message must contain an ID and fields")
        message_id, fields = message
        if not isinstance(fields, Mapping):
            raise ValueError("Redis stream message fields must be a mapping")
        normalized.append(
            (
                _text(message_id),
                {_text(key): _text(value) for key, value in fields.items()},
            )
        )
    return normalized


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)
