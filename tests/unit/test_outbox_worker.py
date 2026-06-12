import asyncio

import pytest

from app.workers import outbox_publisher


@pytest.mark.asyncio
async def test_outbox_worker_retries_after_infrastructure_error(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    class FakePublisher:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def publish_once(self) -> int:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConnectionError("database unavailable")
            raise asyncio.CancelledError

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(outbox_publisher, "OutboxPublisher", FakePublisher)
    monkeypatch.setattr(outbox_publisher, "get_session_factory", lambda: object())
    monkeypatch.setattr(outbox_publisher, "get_redis", lambda: object())
    monkeypatch.setattr(outbox_publisher.asyncio, "sleep", no_sleep)

    with pytest.raises(asyncio.CancelledError):
        await outbox_publisher.run()

    assert attempts == 2
