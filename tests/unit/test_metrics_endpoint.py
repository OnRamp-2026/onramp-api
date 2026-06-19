from __future__ import annotations

from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_db_session
from app.config import get_settings
from app.services.prometheus_metrics import StreamGroupMetric, WorkerMetricSnapshot


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEBUG", "false")
    get_settings.cache_clear()


@pytest.fixture
def metrics_app() -> object:
    from app.main import create_app

    app = create_app()

    async def fake_session():
        yield object()

    app.dependency_overrides[get_db_session] = fake_session
    return app


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_prometheus_text(metrics_app, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_collect(*_args: object, **_kwargs: object) -> WorkerMetricSnapshot:
        return WorkerMetricSnapshot(
            collected_at=datetime(2026, 6, 19, tzinfo=UTC),
            report_jobs_queued=2,
            report_jobs_processing=1,
            event_outbox_pending=3,
            stream_lengths={"onramp:stt:completed:v1": 4},
            stream_groups=[
                StreamGroupMetric(
                    stream="onramp:stt:completed:v1",
                    group="report-workers",
                    pending=1,
                    lag=2,
                )
            ],
        )

    monkeypatch.setattr("app.api.metrics.collect_worker_metric_snapshot", fake_collect)
    monkeypatch.setattr("app.api.metrics.get_redis", lambda: object())

    transport = ASGITransport(app=metrics_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
    assert 'onramp_report_jobs{status="queued"} 2' in response.text
    assert 'onramp_redis_stream_group_lag{stream="onramp:stt:completed:v1",group="report-workers"} 2' in response.text
