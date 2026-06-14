from __future__ import annotations

from typing import Any, cast

import pytest
from redis.asyncio import Redis

from app.queue.consumer import read_new_or_reclaimed


class FakeRedis:
    def __init__(self, response: Any) -> None:
        self.response = response

    async def xautoclaim(self, *args: Any, **kwargs: Any) -> tuple[str, list[Any]]:
        return ("0-0", [])

    async def xreadgroup(self, *args: Any, **kwargs: Any) -> Any:
        return self.response


async def read_messages(response: Any) -> list[tuple[str, dict[str, str]]]:
    return await read_new_or_reclaimed(
        cast(Redis, FakeRedis(response)),
        stream="transcription-events",
        group="onramp-api",
        consumer="worker-1",
        block_ms=100,
        count=10,
        reclaim_idle_ms=60_000,
    )


@pytest.mark.asyncio
async def test_read_new_messages_from_list_response() -> None:
    response = [
        (
            b"transcription-events",
            [(b"1-0", {b"event_id": b"evt-1", b"event_type": b"transcription.completed"})],
        )
    ]

    messages = await read_messages(response)

    assert messages == [
        (
            "1-0",
            {"event_id": "evt-1", "event_type": "transcription.completed"},
        )
    ]


@pytest.mark.asyncio
async def test_read_new_messages_from_mapping_response() -> None:
    response = {
        "transcription-events": [
            ("2-0", {"event_id": "evt-2", "event_type": "transcription.progressed"}),
        ]
    }

    messages = await read_messages(response)

    assert messages == [
        (
            "2-0",
            {"event_id": "evt-2", "event_type": "transcription.progressed"},
        )
    ]


@pytest.mark.asyncio
async def test_unrecoverable_message_is_acknowledged() -> None:
    from app.workers.stt_event_consumer import process_message

    class AckRedis:
        def __init__(self) -> None:
            self.acked: list[tuple[str, str, str]] = []

        async def xack(self, stream: str, group: str, message_id: str) -> None:
            self.acked.append((stream, group, message_id))

    class InvalidService:
        async def process(self, envelope: object) -> None:
            raise AssertionError("decode should fail first")

    redis = AckRedis()
    await process_message(
        cast(Redis, redis),
        InvalidService(),  # type: ignore[arg-type]
        stream="onramp:stt:progress:v1",
        group="onramp-workflow-updaters",
        message_id="1-0",
        fields={"event_type": "missing-required-fields"},
    )

    assert redis.acked == [("onramp:stt:progress:v1", "onramp-workflow-updaters", "1-0")]
