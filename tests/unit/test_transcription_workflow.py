from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import UUID

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models import TranscriptionWorkflow, WorkflowStatus
from app.models.transcription import TranscriptionCreateRequest, UploadCompleteRequest
from app.services.stt_result_client import (
    SttCompleteUploadResponse,
    SttCreateTranscriptionResponse,
    SttUploadInstruction,
)
from app.services.transcription_service import (
    TranscriptionConflictError,
    TranscriptionNotFoundError,
    complete_upload,
    create_workflow,
    get_workflow,
    status_response,
)


class FakeSttResultClient:
    def __init__(self, *, fail_complete: bool = False) -> None:
        self.create_calls: int = 0
        self.complete_calls: int = 0
        self.fail_complete = fail_complete

    async def create_transcription(
        self,
        *,
        tenant_id: str,
        transcription_id: UUID,
        filename: str,
        content_type: str,
        size_bytes: int,
        idempotency_key: str | None = None,
    ) -> SttCreateTranscriptionResponse:
        self.create_calls += 1
        object_key = f"tenants/{tenant_id}/transcriptions/{transcription_id}/source/{filename}"
        return SttCreateTranscriptionResponse(
            transcription_id=transcription_id,
            status="awaiting_upload",
            source_object_key=object_key,
            upload=SttUploadInstruction(
                url=f"https://storage.test/{object_key}",
                method="PUT",
                headers={"Content-Type": content_type},
                expires_at=datetime.now(UTC) + timedelta(seconds=900),
            ),
        )

    async def complete_upload(
        self,
        transcription_id: UUID,
        *,
        etag: str | None,
        size_bytes: int,
    ) -> SttCompleteUploadResponse:
        if self.fail_complete:
            raise httpx.HTTPStatusError(
                "409 Conflict",
                request=httpx.Request("POST", "http://stt"),
                response=httpx.Response(409),
            )
        self.complete_calls += 1
        return SttCompleteUploadResponse(transcription_id=transcription_id, status="queued")


@pytest.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


def create_request() -> TranscriptionCreateRequest:
    return TranscriptionCreateRequest(
        filename="장애 대응 회의.m4a",
        content_type="audio/mp4",
        size_bytes=1024,
        title="장애 대응 회의",
        language="ko-KR",
        category="장애대응",
    )


@pytest.mark.asyncio
async def test_create_workflow_is_idempotent_per_tenant(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    stt_client = FakeSttResultClient()

    async with session_factory() as session:
        first, first_created = await create_workflow(
            session,
            stt_client,
            tenant_id="tenant-a",
            idempotency_key="same-key",
            request=create_request(),
        )
        await session.commit()

    async with session_factory() as session:
        second, second_created = await create_workflow(
            session,
            stt_client,
            tenant_id="tenant-a",
            idempotency_key="same-key",
            request=create_request(),
        )

    assert first_created is True
    assert second_created is False
    assert second.workflow.id == first.workflow.id
    assert second.workflow.transcription_id == first.workflow.transcription_id
    assert stt_client.create_calls == 2


@pytest.mark.asyncio
async def test_same_idempotency_key_isolated_by_tenant(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    stt_client = FakeSttResultClient()

    async with session_factory() as session:
        first, _ = await create_workflow(
            session,
            stt_client,
            tenant_id="tenant-a",
            idempotency_key="same-key",
            request=create_request(),
        )
        await session.commit()

    async with session_factory() as session:
        second, created = await create_workflow(
            session,
            stt_client,
            tenant_id="tenant-b",
            idempotency_key="same-key",
            request=create_request(),
        )

    assert created is True
    assert second.workflow.id != first.workflow.id


@pytest.mark.asyncio
async def test_complete_upload_queues_workflow(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    stt_client = FakeSttResultClient()
    async with session_factory() as session:
        created, _ = await create_workflow(
            session,
            stt_client,
            tenant_id="tenant-a",
            idempotency_key=None,
            request=create_request(),
        )
        await session.commit()

    async with session_factory() as session:
        workflow = await complete_upload(
            session,
            stt_client,
            tenant_id="tenant-a",
            transcription_id=created.workflow.transcription_id,
            request=UploadCompleteRequest(etag='"abc123"', size_bytes=1024),
        )
        await session.commit()

    async with session_factory() as session:
        persisted = await session.scalar(select(TranscriptionWorkflow).where(TranscriptionWorkflow.id == workflow.id))

    assert persisted is not None
    assert persisted.status == WorkflowStatus.queued
    assert persisted.source_etag == '"abc123"'
    assert stt_client.complete_calls == 1


@pytest.mark.asyncio
async def test_complete_upload_is_idempotent_when_already_queued(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    stt_client = FakeSttResultClient()
    async with session_factory() as session:
        created, _ = await create_workflow(
            session,
            stt_client,
            tenant_id="tenant-a",
            idempotency_key=None,
            request=create_request(),
        )
        await session.commit()

    complete_request = UploadCompleteRequest(etag='"abc123"', size_bytes=1024)

    async with session_factory() as session:
        await complete_upload(
            session,
            stt_client,
            tenant_id="tenant-a",
            transcription_id=created.workflow.transcription_id,
            request=complete_request,
        )
        await session.commit()

    async with session_factory() as session:
        workflow = await complete_upload(
            session,
            stt_client,
            tenant_id="tenant-a",
            transcription_id=created.workflow.transcription_id,
            request=complete_request,
        )
        await session.commit()

    assert workflow.status == WorkflowStatus.queued
    assert stt_client.complete_calls == 1  # second call skipped (already queued)


@pytest.mark.asyncio
async def test_complete_upload_rollback_keeps_workflow_awaiting(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    stt_client = FakeSttResultClient()
    async with session_factory() as session:
        created, _ = await create_workflow(
            session,
            stt_client,
            tenant_id="tenant-a",
            idempotency_key=None,
            request=create_request(),
        )
        await session.commit()

    async with session_factory() as session:
        await complete_upload(
            session,
            stt_client,
            tenant_id="tenant-a",
            transcription_id=created.workflow.transcription_id,
            request=UploadCompleteRequest(etag='"abc123"', size_bytes=1024),
        )
        await session.rollback()

    async with session_factory() as session:
        workflow = await session.scalar(
            select(TranscriptionWorkflow).where(
                TranscriptionWorkflow.transcription_id == created.workflow.transcription_id
            )
        )

    assert workflow is not None
    assert workflow.status == WorkflowStatus.awaiting_upload


@pytest.mark.asyncio
async def test_complete_upload_raises_conflict_when_stt_returns_409(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    stt_client_ok = FakeSttResultClient()
    async with session_factory() as session:
        created, _ = await create_workflow(
            session,
            stt_client_ok,
            tenant_id="tenant-a",
            idempotency_key=None,
            request=create_request(),
        )
        await session.commit()

    stt_client_fail = FakeSttResultClient(fail_complete=True)
    async with session_factory() as session:
        with pytest.raises(TranscriptionConflictError):
            await complete_upload(
                session,
                stt_client_fail,
                tenant_id="tenant-a",
                transcription_id=created.workflow.transcription_id,
                request=UploadCompleteRequest(etag='"abc123"', size_bytes=1024),
            )


@pytest.mark.asyncio
async def test_get_workflow_enforces_tenant_ownership(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    stt_client = FakeSttResultClient()
    async with session_factory() as session:
        created, _ = await create_workflow(
            session,
            stt_client,
            tenant_id="tenant-a",
            idempotency_key=None,
            request=create_request(),
        )
        await session.commit()

    async with session_factory() as session:
        with pytest.raises(TranscriptionNotFoundError):
            await get_workflow(
                session,
                tenant_id="tenant-b",
                transcription_id=UUID(str(created.workflow.transcription_id)),
            )


def test_report_failed_status_is_normalized() -> None:
    workflow = TranscriptionWorkflow(
        transcription_id=UUID("00000000-0000-0000-0000-000000000001"),
        tenant_id="tenant-a",
        source_object_key="source",
        source_filename="meeting.m4a",
        source_content_type="audio/mp4",
        source_size_bytes=1024,
        title="장애 대응 회의",
        language="ko-KR",
        category="장애대응",
        status=WorkflowStatus.report_failed,
        total_chunks=0,
        completed_chunks=0,
        failed_chunks=0,
        updated_at=datetime.now(UTC),
    )

    assert status_response(workflow).report.status == "failed"
