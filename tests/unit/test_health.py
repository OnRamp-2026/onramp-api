from unittest.mock import AsyncMock

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
async def test_health_ready_returns_ok_when_all_dependencies_are_available(client, monkeypatch: pytest.MonkeyPatch):
    """상세 헬스체크 — 모든 의존성이 정상이면 ok."""
    monkeypatch.setattr("app.api.v1.health.check_qdrant", AsyncMock(return_value=True))
    monkeypatch.setattr("app.api.v1.health.check_postgres", AsyncMock(return_value=True))
    monkeypatch.setattr("app.api.v1.health.check_redis", AsyncMock(return_value=True))

    response = await client.get("/v1/health/ready")
    assert response.status_code == 200

    data = response.json()
    assert data == {
        "status": "ok",
        "checks": {
            "qdrant": "ok",
            "postgres": "ok",
            "redis": "ok",
        },
    }


@pytest.mark.asyncio
async def test_health_ready_returns_503_when_any_dependency_is_unavailable(client, monkeypatch: pytest.MonkeyPatch):
    """상세 헬스체크 — 하나라도 실패하면 degraded + 503."""
    monkeypatch.setattr("app.api.v1.health.check_qdrant", AsyncMock(return_value=True))
    monkeypatch.setattr("app.api.v1.health.check_postgres", AsyncMock(return_value=False))
    monkeypatch.setattr("app.api.v1.health.check_redis", AsyncMock(return_value=True))

    response = await client.get("/v1/health/ready")
    assert response.status_code == 503

    data = response.json()
    assert data == {
        "status": "degraded",
        "checks": {
            "qdrant": "ok",
            "postgres": "error",
            "redis": "ok",
        },
    }
