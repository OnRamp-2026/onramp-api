from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models import EventOutbox, TranscriptionWorkflow, WorkflowStatus
from app.models.transcription import TranscriptionCreateRequest, UploadCompleteRequest
from app.services.transcription_service import (
    TranscriptionConflictError,
    TranscriptionNotFoundError,
    complete_upload,
    create_workflow,
    get_workflow,
)
from app.storage.base import ObjectMetadata, PresignedUpload


class FakeObjectStorage:
    def __init__(self) -> None:
        self.objects: dict[str, ObjectMetadata] = {}
        self.presigned_keys: list[str] = []

    async def create_presigned_upload(
        self,
        object_key: str,
        *,
        content_type: str,
        expires_in_seconds: int,
    ) -> PresignedUpload:
        self.presigned_keys.append(object_key)
        return PresignedUpload(
            method="PUT",
            url=f"https://storage.test/{object_key}",
            headers={"Content-Type": content_type},
            expires_at=datetime.now(UTC) + timedelta(seconds=expires_in_seconds),
        )

    async def head(self, object_key: str) -> ObjectMetadata:
        return self.objects[object_key]


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
    storage = FakeObjectStorage()

    async with session_factory() as session:
        first, first_created = await create_workflow(
            session,
            storage,
            tenant_id="tenant-a",
            idempotency_key="same-key",
            request=create_request(),
            upload_ttl_seconds=900,
        )
        await session.commit()

    async with session_factory() as session:
        second, second_created = await create_workflow(
            session,
            storage,
            tenant_id="tenant-a",
            idempotency_key="same-key",
            request=create_request(),
            upload_ttl_seconds=900,
        )

    assert first_created is True
    assert second_created is False
    assert second.workflow.id == first.workflow.id
    assert second.workflow.transcription_id == first.workflow.transcription_id
    assert len(storage.presigned_keys) == 2


@pytest.mark.asyncio
async def test_same_idempotency_key_isolated_by_tenant(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    storage = FakeObjectStorage()

    async with session_factory() as session:
        first, _ = await create_workflow(
            session,
            storage,
            tenant_id="tenant-a",
            idempotency_key="same-key",
            request=create_request(),
            upload_ttl_seconds=900,
        )
        await session.commit()

    async with session_factory() as session:
        second, created = await create_workflow(
            session,
            storage,
            tenant_id="tenant-b",
            idempotency_key="same-key",
            request=create_request(),
            upload_ttl_seconds=900,
        )

    assert created is True
    assert second.workflow.id != first.workflow.id


@pytest.mark.asyncio
async def test_complete_upload_queues_workflow_and_stores_outbox(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    storage = FakeObjectStorage()
    async with session_factory() as session:
        created, _ = await create_workflow(
            session,
            storage,
            tenant_id="tenant-a",
            idempotency_key=None,
            request=create_request(),
            upload_ttl_seconds=900,
        )
        await session.commit()

    storage.objects[created.workflow.source_object_key] = ObjectMetadata(
        object_key=created.workflow.source_object_key,
        size_bytes=1024,
        content_type="audio/mp4",
        etag='"abc123"',
    )

    async with session_factory() as session:
        workflow = await complete_upload(
            session,
            storage,
            tenant_id="tenant-a",
            transcription_id=created.workflow.transcription_id,
            request=UploadCompleteRequest(etag='"abc123"', size_bytes=1024),
        )
        await session.commit()

    async with session_factory() as session:
        outbox = await session.scalar(select(EventOutbox))
        persisted = await session.scalar(select(TranscriptionWorkflow).where(TranscriptionWorkflow.id == workflow.id))

    assert persisted is not None
    assert persisted.status == WorkflowStatus.queued
    assert persisted.source_etag == '"abc123"'
    assert outbox is not None
    assert outbox.event_type == "transcription.requested"
    assert outbox.stream_name == "onramp:stt:requests:v1"
    assert outbox.payload_json["transcription_id"] == str(workflow.transcription_id)
    assert outbox.payload_json["tenant_id"] == "tenant-a"


@pytest.mark.asyncio
async def test_complete_upload_does_not_duplicate_outbox(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    storage = FakeObjectStorage()
    async with session_factory() as session:
        created, _ = await create_workflow(
            session,
            storage,
            tenant_id="tenant-a",
            idempotency_key=None,
            request=create_request(),
            upload_ttl_seconds=900,
        )
        await session.commit()

    storage.objects[created.workflow.source_object_key] = ObjectMetadata(
        object_key=created.workflow.source_object_key,
        size_bytes=1024,
        content_type="audio/mp4",
        etag='"abc123"',
    )
    complete_request = UploadCompleteRequest(etag='"abc123"', size_bytes=1024)

    async with session_factory() as session:
        await complete_upload(
            session,
            storage,
            tenant_id="tenant-a",
            transcription_id=created.workflow.transcription_id,
            request=complete_request,
        )
        await session.commit()

    async with session_factory() as session:
        workflow = await complete_upload(
            session,
            storage,
            tenant_id="tenant-a",
            transcription_id=created.workflow.transcription_id,
            request=complete_request,
        )
        await session.commit()
        outbox_count = await session.scalar(select(func.count()).select_from(EventOutbox))

    assert workflow.status == WorkflowStatus.queued
    assert outbox_count == 1


@pytest.mark.asyncio
async def test_complete_upload_rollback_keeps_workflow_and_outbox_atomic(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    storage = FakeObjectStorage()
    async with session_factory() as session:
        created, _ = await create_workflow(
            session,
            storage,
            tenant_id="tenant-a",
            idempotency_key=None,
            request=create_request(),
            upload_ttl_seconds=900,
        )
        await session.commit()

    storage.objects[created.workflow.source_object_key] = ObjectMetadata(
        object_key=created.workflow.source_object_key,
        size_bytes=1024,
        content_type="audio/mp4",
        etag='"abc123"',
    )

    async with session_factory() as session:
        await complete_upload(
            session,
            storage,
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
        outbox_count = await session.scalar(select(func.count()).select_from(EventOutbox))

    assert workflow is not None
    assert workflow.status == WorkflowStatus.awaiting_upload
    assert outbox_count == 0


@pytest.mark.asyncio
async def test_complete_upload_rejects_metadata_mismatch(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    storage = FakeObjectStorage()
    async with session_factory() as session:
        created, _ = await create_workflow(
            session,
            storage,
            tenant_id="tenant-a",
            idempotency_key=None,
            request=create_request(),
            upload_ttl_seconds=900,
        )
        await session.commit()

    storage.objects[created.workflow.source_object_key] = ObjectMetadata(
        object_key=created.workflow.source_object_key,
        size_bytes=999,
        content_type="audio/mp4",
        etag='"abc123"',
    )

    async with session_factory() as session:
        with pytest.raises(TranscriptionConflictError):
            await complete_upload(
                session,
                storage,
                tenant_id="tenant-a",
                transcription_id=created.workflow.transcription_id,
                request=UploadCompleteRequest(etag='"abc123"', size_bytes=1024),
            )


@pytest.mark.asyncio
async def test_get_workflow_enforces_tenant_ownership(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    storage = FakeObjectStorage()
    async with session_factory() as session:
        created, _ = await create_workflow(
            session,
            storage,
            tenant_id="tenant-a",
            idempotency_key=None,
            request=create_request(),
            upload_ttl_seconds=900,
        )
        await session.commit()

    async with session_factory() as session:
        with pytest.raises(TranscriptionNotFoundError):
            await get_workflow(
                session,
                tenant_id="tenant-b",
                transcription_id=UUID(str(created.workflow.transcription_id)),
            )
