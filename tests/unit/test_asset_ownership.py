from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models import Report, ReportStatus, TranscriptionWorkflow, WorkflowStatus
from app.middleware.error_handler import OnRampError
from app.services.report_service import get_report
from app.services.transcription_service import TranscriptionNotFoundError, get_workflow


@pytest.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_workflow_and_report_are_scoped_to_current_user(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    transcription_id = uuid4()
    workflow = TranscriptionWorkflow(
        transcription_id=transcription_id,
        tenant_id="tenant-a",
        created_by_user_id="user-a",
        idempotency_key=None,
        status=WorkflowStatus.draft,
        source_object_key="source.m4a",
        source_filename="source.m4a",
        source_content_type="audio/mp4",
        source_size_bytes=1024,
        title="장애 회의",
        language="ko-KR",
        category="장애대응",
    )
    report = Report(
        tenant_id="tenant-a",
        source_transcription_id=transcription_id,
        title="장애 보고서",
        category="장애대응",
        situation="상황",
        cause="원인",
        evidence="근거",
        solution="해결",
        infra_context="환경",
        status=ReportStatus.draft,
        raw_text_sha256="a" * 64,
        corrected_text_sha256="b" * 64,
        dictionary_version="2026-06-21",
        result_object_key="result.json",
    )
    async with session_factory() as session:
        session.add_all([workflow, report])
        await session.commit()

    async with session_factory() as session:
        with pytest.raises(TranscriptionNotFoundError):
            await get_workflow(
                session,
                tenant_id="tenant-a",
                user_id="user-b",
                transcription_id=transcription_id,
            )
        with pytest.raises(OnRampError):
            await get_report(
                session,
                tenant_id="tenant-a",
                user_id="user-b",
                report_id=report.id,
            )
