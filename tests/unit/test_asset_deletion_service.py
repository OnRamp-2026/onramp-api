from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models import EventOutbox, Report, ReportStatus, TranscriptionWorkflow, WorkflowStatus
from app.middleware.error_handler import OnRampError
from app.queue.constants import DELETE_REQUESTED_EVENT_TYPE, STT_REQUEST_STREAM
from app.services.asset_deletion_service import request_asset_deletion


@pytest.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


def draft_workflow(*, user_id: str = "user-a") -> TranscriptionWorkflow:
    return TranscriptionWorkflow(
        transcription_id=uuid4(),
        tenant_id="tenant-a",
        created_by_user_id=user_id,
        status=WorkflowStatus.draft,
        source_object_key="source.m4a",
        source_filename="source.m4a",
        source_content_type="audio/mp4",
        source_size_bytes=1024,
        title="장애 회의",
        language="ko-KR",
        category="장애대응",
    )


def report_for(workflow: TranscriptionWorkflow, *, status: ReportStatus = ReportStatus.draft) -> Report:
    return Report(
        tenant_id=workflow.tenant_id,
        source_transcription_id=workflow.transcription_id,
        title=workflow.title,
        category=workflow.category,
        situation="상황",
        cause="원인",
        evidence="근거",
        solution="해결",
        infra_context="환경",
        status=status,
        raw_text_sha256="a" * 64,
        corrected_text_sha256="b" * 64,
        dictionary_version="2026-06-21",
        result_object_key="result.json",
    )


@pytest.mark.asyncio
async def test_request_deletion_marks_draft_and_emits_outbox_once(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    workflow = draft_workflow()
    async with session_factory() as session:
        session.add_all([workflow, report_for(workflow)])
        await session.commit()

    async with session_factory() as session:
        first = await request_asset_deletion(
            session,
            tenant_id="tenant-a",
            user_id="user-a",
            transcription_id=workflow.transcription_id,
        )
        await session.commit()
        second = await request_asset_deletion(
            session,
            tenant_id="tenant-a",
            user_id="user-a",
            transcription_id=workflow.transcription_id,
        )
        await session.commit()

    assert first.status == "deleting"
    assert second.status == "deleting"
    async with session_factory() as session:
        persisted = await session.get(TranscriptionWorkflow, workflow.id)
        events = list(await session.scalars(select(EventOutbox)))
    assert persisted is not None
    assert persisted.status == WorkflowStatus.deleting
    assert len(events) == 1
    assert events[0].event_type == DELETE_REQUESTED_EVENT_TYPE
    assert events[0].stream_name == STT_REQUEST_STREAM
    assert events[0].payload_json == {
        "transcription_id": str(workflow.transcription_id),
        "tenant_id": "tenant-a",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("workflow_status", "report_status"),
    [
        (WorkflowStatus.transcribing, None),
        (WorkflowStatus.published, ReportStatus.published),
        (WorkflowStatus.draft, ReportStatus.publishing),
    ],
)
async def test_request_deletion_rejects_non_draft_assets(
    session_factory: async_sessionmaker[AsyncSession],
    workflow_status: WorkflowStatus,
    report_status: ReportStatus | None,
) -> None:
    workflow = draft_workflow()
    workflow.status = workflow_status
    async with session_factory() as session:
        session.add(workflow)
        if report_status is not None:
            session.add(report_for(workflow, status=report_status))
        await session.commit()

    async with session_factory() as session:
        with pytest.raises(OnRampError) as exc_info:
            await request_asset_deletion(
                session,
                tenant_id="tenant-a",
                user_id="user-a",
                transcription_id=workflow.transcription_id,
            )

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_request_deletion_hides_other_users_asset(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    workflow = draft_workflow(user_id="user-b")
    async with session_factory() as session:
        session.add_all([workflow, report_for(workflow)])
        await session.commit()

    async with session_factory() as session:
        with pytest.raises(OnRampError) as exc_info:
            await request_asset_deletion(
                session,
                tenant_id="tenant-a",
                user_id="user-a",
                transcription_id=workflow.transcription_id,
            )

    assert exc_info.value.status_code == 404
