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
        )
    return _client


async def check_redis() -> bool:
    """Redis 연결 상태 확인."""
    try:
        client = get_redis()
        result = await client.ping()
        return bool(result)
    except Exception:
        return False


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None
