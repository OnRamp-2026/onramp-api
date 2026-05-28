from qdrant_client import QdrantClient

from app.config import get_settings

_client: QdrantClient | None = None


def get_qdrant() -> QdrantClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
    return _client


async def check_qdrant() -> bool:
    """Qdrant 연결 상태 확인."""
    try:
        client = get_qdrant()
        client.get_collections()
        return True
    except Exception:
        return False


def close_qdrant() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
