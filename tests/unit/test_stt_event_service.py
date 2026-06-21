from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models import Report, ReportJob, ReportJobStatus, ReportStatus, TranscriptionWorkflow, WorkflowStatus
from app.queue.events import ProgressUpdated, StreamEnvelope, TranscriptCompleted, TranscriptionCompleted
from app.services.stt_event_service import SttEventService, UnrecoverableSttEventError


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

    with pytest.raises(UnrecoverableSttEventError, match="tenant"):
        await service.process(envelope)


@pytest.mark.asyncio
async def test_event_for_unknown_workflow_is_unrecoverable(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    service = SttEventService(session_factory)
    envelope = StreamEnvelope(
        event_id="evt-unknown-workflow",
        event_type="transcription.completed",
        payload=TranscriptionCompleted(
            transcription_id=uuid4(),
            tenant_id="tenant-a",
            raw_text_sha256="a" * 64,
            corrected_text_sha256="b" * 64,
            dictionary_version="2026-06-14",
            result_object_key="tenants/tenant-a/result.json",
            completed_at=datetime.now(UTC),
        ).model_dump(mode="json"),
    )

    with pytest.raises(UnrecoverableSttEventError, match="workflow"):
        await service.process(envelope)


@pytest.mark.asyncio
async def test_late_progress_event_does_not_regress_report_workflow(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    workflow = await create_workflow(session_factory)
    async with session_factory() as session:
        persisted = await session.get(TranscriptionWorkflow, workflow.id)
        assert persisted is not None
        persisted.status = WorkflowStatus.draft
        await session.commit()

    service = SttEventService(session_factory)
    envelope = StreamEnvelope(
        event_id="evt-late-progress",
        event_type="transcription.progressed",
        payload=ProgressUpdated(
            transcription_id=workflow.transcription_id,
            tenant_id="tenant-a",
            status="transcribing",
            completed_chunks=9,
            total_chunks=10,
            failed_chunks=0,
            progress_ratio=0.9,
            occurred_at=datetime.now(UTC),
        ).model_dump(mode="json"),
    )

    await service.process(envelope)

    async with session_factory() as session:
        persisted = await session.get(TranscriptionWorkflow, workflow.id)
    assert persisted is not None
    assert persisted.status == WorkflowStatus.draft


@pytest.mark.asyncio
async def test_duplicate_completion_event_does_not_regress_completed_report(
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

    await service.process(completion("evt-initial-completion"))
    async with session_factory() as session:
        persisted = await session.get(TranscriptionWorkflow, workflow.id)
        assert persisted is not None
        persisted.status = WorkflowStatus.draft
        await session.commit()

    await service.process(completion("evt-duplicate-completion"))

    async with session_factory() as session:
        persisted = await session.get(TranscriptionWorkflow, workflow.id)
    assert persisted is not None
    assert persisted.status == WorkflowStatus.draft


@pytest.mark.asyncio
async def test_different_stream_groups_can_process_same_event_id(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    workflow = await create_workflow(session_factory)
    service = SttEventService(session_factory)
    event_id = "evt-shared-across-streams"

    await service.process(
        StreamEnvelope(
            event_id=event_id,
            event_type="transcription.progressed",
            payload=ProgressUpdated(
                transcription_id=workflow.transcription_id,
                tenant_id="tenant-a",
                status="merging",
                completed_chunks=10,
                total_chunks=10,
                failed_chunks=0,
                progress_ratio=1,
                occurred_at=datetime.now(UTC),
            ).model_dump(mode="json"),
        )
    )
    await service.process(
        StreamEnvelope(
            event_id=event_id,
            event_type="transcription.transcript.completed",
            payload=TranscriptCompleted(
                transcription_id=workflow.transcription_id,
                tenant_id="tenant-a",
                result_object_key="tenants/tenant-a/transcript.json",
            ).model_dump(mode="json"),
        )
    )

    async with session_factory() as session:
        persisted = await session.get(TranscriptionWorkflow, workflow.id)
    assert persisted is not None
    assert persisted.status == WorkflowStatus.transcript_completed
    assert persisted.transcript_completed_received_at is not None


@pytest.mark.asyncio
async def test_deleted_event_removes_deleting_workflow_and_related_rows(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    workflow = await create_workflow(session_factory)
    async with session_factory() as session:
        persisted = await session.get(TranscriptionWorkflow, workflow.id)
        assert persisted is not None
        persisted.status = WorkflowStatus.deleting
        session.add(
            ReportJob(
                tenant_id=persisted.tenant_id,
                source_transcription_id=persisted.transcription_id,
                status=ReportJobStatus.completed,
                raw_text_sha256="a" * 64,
                corrected_text_sha256="b" * 64,
                dictionary_version="v1",
                result_object_key="result.json",
            )
        )
        session.add(
            Report(
                tenant_id=persisted.tenant_id,
                source_transcription_id=persisted.transcription_id,
                title=persisted.title,
                category=persisted.category,
                situation="상황",
                cause="원인",
                evidence="근거",
                solution="해결",
                infra_context="환경",
                status=ReportStatus.draft,
                raw_text_sha256="a" * 64,
                corrected_text_sha256="b" * 64,
                dictionary_version="v1",
                result_object_key="result.json",
            )
        )
        await session.commit()

    envelope = StreamEnvelope(
        event_id="evt-deleted",
        event_type="transcription.deleted",
        payload={
            "transcription_id": str(workflow.transcription_id),
            "tenant_id": workflow.tenant_id,
        },
    )
    service = SttEventService(session_factory)
    await service.process(envelope)
    await service.process(envelope)

    async with session_factory() as session:
        assert await session.get(TranscriptionWorkflow, workflow.id) is None
        assert list(await session.scalars(select(ReportJob))) == []
        assert list(await session.scalars(select(Report))) == []


@pytest.mark.asyncio
async def test_progress_after_deletion_request_does_not_regress_workflow(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    workflow = await create_workflow(session_factory)
    async with session_factory() as session:
        persisted = await session.get(TranscriptionWorkflow, workflow.id)
        assert persisted is not None
        persisted.status = WorkflowStatus.deleting
        await session.commit()

    service = SttEventService(session_factory)
    envelope = StreamEnvelope(
        event_id="evt-progress-after-delete",
        event_type="transcription.progressed",
        payload=ProgressUpdated(
            transcription_id=workflow.transcription_id,
            tenant_id="tenant-a",
            status="correcting",
            completed_chunks=10,
            total_chunks=10,
            failed_chunks=0,
            progress_ratio=1,
            occurred_at=datetime.now(UTC),
        ).model_dump(mode="json"),
    )
    await service.process(envelope)

    async with session_factory() as session:
        persisted = await session.get(TranscriptionWorkflow, workflow.id)
    assert persisted is not None
    assert persisted.status == WorkflowStatus.deleting
