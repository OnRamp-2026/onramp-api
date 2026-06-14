from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models import Report, ReportJob, ReportJobStatus, TranscriptionWorkflow, WorkflowStatus, utcnow
from app.models.response import FiveElementsResponse
from app.services.report_worker import GeneratedReport, ReportWorker
from app.services.stt_result_client import (
    CorrectedTranscriptResult,
    SttResult,
    TranscriptResult,
)


@pytest.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


class FakeSttClient:
    def __init__(self, result: SttResult) -> None:
        self.result = result

    async def get_result(self, transcription_id):  # type: ignore[no-untyped-def]
        return self.result


async def fake_generator(transcript: str, category: str, title: str) -> GeneratedReport:
    assert transcript == "교정된 전사문"
    return GeneratedReport(
        title=title,
        report=FiveElementsResponse(
            situation="상황",
            cause="원인",
            evidence="근거",
            solution="해결",
            infra_context="환경",
        ),
    )


async def create_job(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[TranscriptionWorkflow, ReportJob]:
    transcription_id = uuid4()
    workflow = TranscriptionWorkflow(
        transcription_id=transcription_id,
        tenant_id="tenant-a",
        status=WorkflowStatus.report_queued,
        source_object_key="tenants/tenant-a/source.m4a",
        source_filename="source.m4a",
        source_content_type="audio/mp4",
        source_size_bytes=1024,
        title="장애 대응 회의",
        language="ko-KR",
        category="장애대응",
    )
    job = ReportJob(
        tenant_id="tenant-a",
        source_transcription_id=transcription_id,
        status=ReportJobStatus.queued,
        raw_text_sha256="a" * 64,
        corrected_text_sha256="b" * 64,
        dictionary_version="2026-06-14",
        result_object_key="tenants/tenant-a/result.json",
    )
    async with session_factory() as session:
        session.add_all([workflow, job])
        await session.commit()
    return workflow, job


@pytest.mark.asyncio
async def test_report_worker_persists_draft(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    workflow, job = await create_job(session_factory)
    result = SttResult(
        transcription_id=workflow.transcription_id,
        tenant_id="tenant-a",
        provider="clova",
        audio_duration_sec=120.0,
        dictionary_version="2026-06-14",
        raw=TranscriptResult(text_sha256="a" * 64, text="원본", segments=[]),
        corrected=CorrectedTranscriptResult(
            text_sha256="b" * 64,
            text="교정된 전사문",
            segments=[],
            correction_count=1,
            review_candidate_count=0,
        ),
    )
    worker = ReportWorker(session_factory, FakeSttClient(result), fake_generator)  # type: ignore[arg-type]

    processed = await worker.process_next()

    async with session_factory() as session:
        persisted_job = await session.get(ReportJob, job.id)
        report = await session.scalar(select(Report))
        persisted_workflow = await session.get(TranscriptionWorkflow, workflow.id)
    assert processed is True
    assert persisted_job is not None
    assert persisted_job.status == ReportJobStatus.completed
    assert report is not None
    assert report.situation == "상황"
    assert persisted_workflow is not None
    assert persisted_workflow.status == WorkflowStatus.draft
    assert persisted_workflow.report_id == report.id


@pytest.mark.asyncio
async def test_report_worker_marks_integrity_mismatch_failed(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    workflow, job = await create_job(session_factory)
    result = SttResult(
        transcription_id=workflow.transcription_id,
        tenant_id="tenant-a",
        provider="clova",
        audio_duration_sec=120.0,
        dictionary_version="2026-06-14",
        raw=TranscriptResult(text_sha256="f" * 64, text="원본", segments=[]),
        corrected=CorrectedTranscriptResult(
            text_sha256="b" * 64,
            text="교정된 전사문",
            segments=[],
            correction_count=1,
            review_candidate_count=0,
        ),
    )
    worker = ReportWorker(session_factory, FakeSttClient(result), fake_generator)  # type: ignore[arg-type]

    processed = await worker.process_next()

    async with session_factory() as session:
        persisted_job = await session.get(ReportJob, job.id)
        persisted_workflow = await session.get(TranscriptionWorkflow, workflow.id)
    assert processed is True
    assert persisted_job is not None
    assert persisted_job.status == ReportJobStatus.failed
    assert "checksum" in (persisted_job.last_error or "")
    assert persisted_workflow is not None
    assert persisted_workflow.status == WorkflowStatus.report_failed


@pytest.mark.asyncio
async def test_report_worker_reclaims_stale_processing_job(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    workflow, job = await create_job(session_factory)
    async with session_factory() as session:
        persisted_job = await session.get(ReportJob, job.id)
        assert persisted_job is not None
        persisted_job.status = ReportJobStatus.processing
        persisted_job.updated_at = utcnow() - timedelta(minutes=10)
        await session.commit()

    result = SttResult(
        transcription_id=workflow.transcription_id,
        tenant_id="tenant-a",
        provider="clova",
        audio_duration_sec=120.0,
        dictionary_version="2026-06-14",
        raw=TranscriptResult(text_sha256="a" * 64, text="원본", segments=[]),
        corrected=CorrectedTranscriptResult(
            text_sha256="b" * 64,
            text="교정된 전사문",
            segments=[],
            correction_count=1,
            review_candidate_count=0,
        ),
    )
    worker = ReportWorker(
        session_factory,
        FakeSttClient(result),  # type: ignore[arg-type]
        fake_generator,
        processing_timeout_seconds=300,
    )

    processed = await worker.process_next()

    async with session_factory() as session:
        persisted_job = await session.get(ReportJob, job.id)
    assert processed is True
    assert persisted_job is not None
    assert persisted_job.status == ReportJobStatus.completed
    assert persisted_job.retry_count == 1
