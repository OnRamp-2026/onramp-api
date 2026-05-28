import pytest


@pytest.mark.asyncio
async def test_health(client):
    """기본 헬스체크 — 서버가 살아있으면 ok."""
    response = await client.get("/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "OnRamp API"
    assert "version" in data


@pytest.mark.asyncio
async def test_health_ready(client):
    """상세 헬스체크 — DB 연결 상태 확인."""
    response = await client.get("/v1/health/ready")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "checks" in data
    assert "qdrant" in data["checks"]
    assert "postgres" in data["checks"]
    assert "redis" in data["checks"]
