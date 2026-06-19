from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models import EventOutbox, ReportJob, ReportJobStatus
from app.services.prometheus_metrics import collect_worker_metric_snapshot, render_worker_metrics


class FakeRedis:
    def __init__(self) -> None:
        self.lengths = {
            "onramp:stt:progress:v1": 12,
            "onramp:stt:transcript-completed:v1": 7,
            "onramp:stt:completed:v1": 3,
        }
        self.groups = {
            "onramp:stt:progress:v1": [{"name": "onramp-workflow-updaters", "pending": 2, "lag": 4}],
            "onramp:stt:transcript-completed:v1": [{"name": "onramp-transcript-observers", "pending": 1, "lag": 2}],
            "onramp:stt:completed:v1": [{"name": "report-workers", "pending": 5, "lag": 6}],
        }

    async def xlen(self, stream: str) -> int:
        return self.lengths.get(stream, 0)

    async def xinfo_groups(self, stream: str) -> list[dict[str, object]]:
        return self.groups.get(stream, [])


@pytest.mark.asyncio
async def test_collect_worker_metric_snapshot() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        session.add_all(
            [
                ReportJob(
                    tenant_id="tenant1-onramp",
                    source_transcription_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                    status=ReportJobStatus.queued,
                    raw_text_sha256="a",
                    corrected_text_sha256="b",
                    dictionary_version="1",
                    result_object_key="s3://result-a",
                ),
                ReportJob(
                    tenant_id="tenant1-onramp",
                    source_transcription_id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                    status=ReportJobStatus.processing,
                    raw_text_sha256="c",
                    corrected_text_sha256="d",
                    dictionary_version="1",
                    result_object_key="s3://result-b",
                ),
                EventOutbox(
                    id="evt-1",
                    aggregate_type="workflow",
                    aggregate_id="workflow-1",
                    event_type="transcription.completed",
                    stream_name="onramp:stt:completed:v1",
                    payload_json={"ok": True},
                    available_at=datetime.now(UTC) - timedelta(seconds=1),
                ),
            ]
        )
        await session.commit()

    async with session_factory() as session:
        snapshot = await collect_worker_metric_snapshot(session, FakeRedis())

    assert snapshot.report_jobs_queued == 1
    assert snapshot.report_jobs_processing == 1
    assert snapshot.event_outbox_pending == 1
    assert snapshot.stream_lengths["onramp:stt:completed:v1"] == 3
    assert any(metric.group == "report-workers" and metric.lag == 6 for metric in snapshot.stream_groups)

    rendered = render_worker_metrics(snapshot)
    assert 'onramp_report_jobs{status="queued"} 1' in rendered
    assert 'onramp_event_outbox_pending 1' in rendered
    assert 'onramp_redis_stream_group_lag{stream="onramp:stt:completed:v1",group="report-workers"} 6' in rendered

    await engine.dispose()
