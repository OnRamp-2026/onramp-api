from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models import ReportJob, TranscriptionWorkflow, WorkflowStatus
from app.queue.events import ProgressUpdated, StreamEnvelope, TranscriptionCompleted
from app.services.stt_event_service import SttEventService


@pytest.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


async def create_workflow(session_factory: async_sessionmaker[AsyncSession]) -> TranscriptionWorkflow:
    workflow = TranscriptionWorkflow(
        transcription_id=uuid4(),
        tenant_id="tenant-a",
        status=WorkflowStatus.queued,
        source_object_key="tenants/tenant-a/source.m4a",
        source_filename="source.m4a",
        source_content_type="audio/mp4",
        source_size_bytes=1024,
        title="장애 대응 회의",
        language="ko-KR",
        category="장애대응",
    )
    async with session_factory() as session:
        session.add(workflow)
        await session.commit()
    return workflow


@pytest.mark.asyncio
async def test_progress_event_updates_workflow_snapshot(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    workflow = await create_workflow(session_factory)
    service = SttEventService(session_factory)
    envelope = StreamEnvelope(
        event_id="evt-progress",
        event_type="transcription.progressed",
        payload=ProgressUpdated(
            transcription_id=workflow.transcription_id,
            tenant_id="tenant-a",
            status="transcribing",
            completed_chunks=3,
            total_chunks=10,
            failed_chunks=1,
            progress_ratio=0.4,
            occurred_at=datetime.now(UTC),
        ).model_dump(mode="json"),
    )

    await service.process(envelope)

    async with session_factory() as session:
        persisted = await session.get(TranscriptionWorkflow, workflow.id)
    assert persisted is not None
    assert persisted.status == WorkflowStatus.transcribing
    assert persisted.completed_chunks == 3
    assert persisted.failed_chunks == 1
    assert persisted.total_chunks == 10


@pytest.mark.asyncio
async def test_completion_event_creates_one_report_job(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    workflow = await create_workflow(session_factory)
    service = SttEventService(session_factory)

    def completion(event_id: str) -> StreamEnvelope:
        return StreamEnvelope(
            event_id=event_id,
            event_type="transcription.completed",
            payload=TranscriptionCompleted(
                transcription_id=workflow.transcription_id,
                tenant_id="tenant-a",
                raw_text_sha256="a" * 64,
                corrected_text_sha256="b" * 64,
                dictionary_version="2026-06-14",
                result_object_key="tenants/tenant-a/result.json",
                completed_at=datetime.now(UTC),
            ).model_dump(mode="json"),
        )

    await service.process(completion("evt-complete-1"))
    await service.process(completion("evt-complete-2"))

    async with session_factory() as session:
        report_job_count = await session.scalar(select(func.count()).select_from(ReportJob))
        persisted = await session.get(TranscriptionWorkflow, workflow.id)
    assert report_job_count == 1
    assert persisted is not None
    assert persisted.status == WorkflowStatus.report_queued


@pytest.mark.asyncio
async def test_completion_event_rejects_tenant_mismatch(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    workflow = await create_workflow(session_factory)
    service = SttEventService(session_factory)
    envelope = StreamEnvelope(
        event_id="evt-wrong-tenant",
        event_type="transcription.completed",
        payload=TranscriptionCompleted(
            transcription_id=workflow.transcription_id,
            tenant_id="tenant-b",
            raw_text_sha256="a" * 64,
            corrected_text_sha256="b" * 64,
            dictionary_version="2026-06-14",
            result_object_key="tenants/tenant-b/result.json",
            completed_at=datetime.now(UTC),
        ).model_dump(mode="json"),
    )

    with pytest.raises(ValueError, match="tenant"):
        await service.process(envelope)
