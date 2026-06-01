from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from app.config import Settings, get_settings

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


def ensure_collection(client: QdrantClient | None = None, settings: Settings | None = None) -> None:
    """컬렉션 멱등 생성 + domain payload index. 차원 불일치 시 에러."""
    client = client or get_qdrant()
    settings = settings or get_settings()
    name = settings.qdrant_collection

    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        vectors = client.get_collection(name).config.params.vectors
        # 단일 unnamed 벡터만 사용 — VectorParams가 아니면 차원 검사 생략
        current_dim = vectors.size if isinstance(vectors, VectorParams) else settings.embedding_dim
        if current_dim != settings.embedding_dim:
            raise ValueError(
                f"컬렉션 '{name}' 차원 불일치: 기존 {current_dim} != 설정 {settings.embedding_dim} "
                "(임베딩 모델 변경 시 재색인 필요)"
            )
        return

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
    )
    # domain 필터 검색용 keyword index
    client.create_payload_index(name, field_name="domain", field_schema=PayloadSchemaType.KEYWORD)


def close_qdrant() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
