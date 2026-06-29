from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_db_session
from app.api.v1.monitoring import router as monitoring_router
from app.auth.session import SessionUser, get_current_user
from app.db.base import Base
from app.db.models import ChatObservation
from app.middleware.error_handler import register_error_handlers

app = FastAPI()
register_error_handlers(app)
app.include_router(monitoring_router, prefix="/v1")


@pytest.fixture
async def monitoring_client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncClient]:
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("MONITORING_ALLOW_ALL_SCOPE_DEMO", "false")
    from app.config import get_settings

    get_settings.cache_clear()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_session
    app.dependency_overrides[get_current_user] = lambda: SessionUser(
        tenant_id="tenant-a",
        subject="operator",
        provider="test",
        name="Operator",
        email="operator@example.com",
        claims={},
    )

    async with session_factory() as seed_session:
        await _seed_chat_observations(seed_session)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
    get_settings.cache_clear()
    await engine.dispose()


async def _seed_chat_observations(session: AsyncSession) -> None:
    now = datetime.now(UTC)
    rows = [
        _row("tenant-a", "success", now - timedelta(days=5), total_tokens=1000, cost=1.2, duration_ms=900),
        _row(
            "tenant-a", "requery", now - timedelta(days=4), total_tokens=1800, cost=2.4, duration_ms=1800, retry_count=1
        ),
        _row(
            "tenant-a",
            "failure",
            now - timedelta(days=3),
            total_tokens=400,
            cost=0.5,
            duration_ms=2600,
            answerability_status="not_enough_evidence",
        ),
        _row("tenant-a", "success", now - timedelta(days=2), total_tokens=1200, cost=1.4, duration_ms=1100),
        _row("tenant-b", "success", now - timedelta(days=1), total_tokens=900, cost=0.9, duration_ms=950),
        _row("tenant-a", "success", now - timedelta(days=40), total_tokens=900, cost=0.8, duration_ms=700),
    ]
    session.add_all(rows)
    await session.commit()


def _row(
    tenant_id: str,
    result_bucket: str,
    created_at: datetime,
    *,
    total_tokens: int,
    cost: float,
    duration_ms: int,
    retry_count: int = 0,
    answerability_status: str = "answerable",
) -> ChatObservation:
    return ChatObservation(
        request_id=str(uuid4()),
        tenant_id=tenant_id,
        requested_model="gpt-4o-mini",
        model_used="gpt-4o-mini",
        domain="incident",
        answerability_status=answerability_status,
        retry_count=retry_count,
        duration_ms=duration_ms,
        source_count=2,
        result_bucket=result_bucket,
        prompt_tokens=total_tokens // 2,
        completion_tokens=total_tokens // 2,
        total_tokens=total_tokens,
        estimated_cost_usd=cost,
        created_at=created_at,
    )


@pytest.mark.asyncio
async def test_monitoring_overview_returns_five_cards_for_tenant_scope(monitoring_client: AsyncClient) -> None:
    response = await monitoring_client.get("/v1/monitoring/overview", params={"scope": "tenant-a", "period": "30d"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["scope"] == "tenant-a"
    assert len(payload["cards"]) == 5
    assert [card["id"] for card in payload["cards"]] == [
        "token_cost",
        "traffic_usage",
        "response_quality",
        "average_cost",
        "search_quality",
    ]


@pytest.mark.asyncio
async def test_monitoring_all_scope_requires_demo_flag(monitoring_client: AsyncClient) -> None:
    response = await monitoring_client.get("/v1/monitoring/overview", params={"scope": "all", "period": "30d"})

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_response_quality_detail_is_latency_focused(
    monitoring_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MONITORING_ALLOW_ALL_SCOPE_DEMO", "true")
    from app.config import get_settings

    get_settings.cache_clear()
    response = await monitoring_client.get(
        "/v1/monitoring/details/response_quality",
        params={"scope": "tenant-a", "period": "30d"},
    )

    assert response.status_code == 200
    payload = response.json()
    labels = [item["label"] for item in payload["summaryMetrics"]]
    assert "성공률" not in labels
    assert any("p95" in item["label"] for item in payload["breakdownItems"])


@pytest.mark.asyncio
async def test_search_quality_aggregates_success_failure_requery(monitoring_client: AsyncClient) -> None:
    response = await monitoring_client.get(
        "/v1/monitoring/details/search_quality", params={"scope": "tenant-a", "period": "30d"}
    )

    assert response.status_code == 200
    payload = response.json()
    by_label = {item["label"]: item["value"] for item in payload["breakdownItems"]}
    assert by_label["성공 연결"].startswith("50")
    assert by_label["재질의"].startswith("25")
    assert by_label["실패"].startswith("25")
