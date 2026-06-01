"""실 Qdrant 통합 테스트 — make up 상태에서 실행. 미가동 시 skip."""

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from app.config import Settings
from app.db.qdrant import ensure_collection

COLLECTION = "onramp_itest"
DIM = 4


@pytest.fixture
def qclient():
    client = QdrantClient(host="localhost", port=6333)
    try:
        client.get_collections()
    except Exception:
        pytest.skip("Qdrant 미가동 (make up 필요)")
    if COLLECTION in {c.name for c in client.get_collections().collections}:
        client.delete_collection(COLLECTION)
    yield client
    client.delete_collection(COLLECTION)
    client.close()


def test_ensure_collection_idempotent_and_search(qclient):
    settings = Settings(qdrant_collection=COLLECTION, embedding_dim=DIM)
    ensure_collection(client=qclient, settings=settings)
    ensure_collection(client=qclient, settings=settings)  # 재호출 멱등

    qclient.upsert(
        COLLECTION,
        points=[
            PointStruct(id=1, vector=[1, 0, 0, 0], payload={"domain": "장애대응", "content": "DB 복구 절차"}),
            PointStruct(id=2, vector=[0, 1, 0, 0], payload={"domain": "보안규정", "content": "방화벽 정책"}),
        ],
    )

    # dense search — [1,0,0,0]에 가장 가까운 건 id=1
    hits = qclient.query_points(COLLECTION, query=[1, 0, 0, 0], limit=2).points
    assert hits[0].id == 1

    # domain 필터
    filtered = qclient.query_points(
        COLLECTION,
        query=[1, 0, 0, 0],
        limit=2,
        query_filter=Filter(must=[FieldCondition(key="domain", match=MatchValue(value="보안규정"))]),
    ).points
    assert filtered and all(p.payload["domain"] == "보안규정" for p in filtered)


def test_ensure_collection_dim_mismatch_raises(qclient):
    ensure_collection(client=qclient, settings=Settings(qdrant_collection=COLLECTION, embedding_dim=DIM))
    with pytest.raises(ValueError, match="차원 불일치"):
        ensure_collection(client=qclient, settings=Settings(qdrant_collection=COLLECTION, embedding_dim=DIM + 1))
