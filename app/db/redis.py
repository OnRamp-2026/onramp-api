from redis.asyncio import Redis

from app.config import get_settings

_client: Redis | None = None


def get_redis() -> Redis:
    global _client
    if _client is None:
        settings = get_settings()
        _client = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            # XREADGROUP(block=redis_stream_block_ms) 동안 정상적으로 블로킹 대기하는 중에
            # 클라이언트 소켓이 먼저 타임아웃되지 않도록 block 시간보다 충분히 크게 설정.
            socket_timeout=(settings.redis_stream_block_ms / 1000) + 5,
        )
    return _client


async def check_redis() -> bool:
    """Redis 연결 상태 확인."""
    try:
        client = get_redis()
        return bool(await client.ping())  # type: ignore[misc]
    except Exception:
        return False


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None
