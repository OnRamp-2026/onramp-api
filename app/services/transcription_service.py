from __future__ import annotations

import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import PurePath
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import EventOutbox, TranscriptionWorkflow, WorkflowStatus, utcnow
from app.middleware.error_handler import OnRampError
from app.models.transcription import (
    ReportStatus,
    TranscriptionCreateRequest,
    TranscriptionCreateResponse,
    TranscriptionProgress,
    TranscriptionStatusResponse,
    UploadCompleteRequest,
    UploadInstruction,
)
from app.queue.constants import STT_REQUEST_STREAM, TRANSCRIPTION_REQUESTED_EVENT
from app.storage.base import ObjectNotFoundError, ObjectStorage, ObjectStorageError, PresignedUpload


class TranscriptionNotFoundError(OnRampError):
    def __init__(self) -> None:
        super().__init__("전사 workflow를 찾을 수 없습니다.", status_code=404)


class TranscriptionConflictError(OnRampError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=409)


class TranscriptionStorageError(OnRampError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=502)


@dataclass(frozen=True)
class WorkflowCreation:
    workflow: TranscriptionWorkflow
    upload: PresignedUpload | None


def _source_filename(filename: str) -> str:
    basename = PurePath(filename.replace("\\", "/")).name
    return unicodedata.normalize("NFC", basename)[:512]


def _safe_filename(filename: str) -> str:
    normalized = _source_filename(filename)
    safe = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", normalized).strip("._")
    if not safe:
        safe = "audio"
    return safe[:255]


def _object_key(tenant_id: str, transcription_id: UUID, filename: str) -> str:
    return f"tenants/{tenant_id}/transcriptions/{transcription_id}/source/{_safe_filename(filename)}"


def _normalize_content_type(value: str) -> str:
    return value.split(";", 1)[0].strip().lower()


def _normalize_etag(value: str | None) -> str:
    if not value:
        return ""
    normalized = value.strip()
    if normalized.startswith("W/"):
        normalized = normalized[2:]
    return normalized.strip('"')


async def _presign(
    storage: ObjectStorage,
    workflow: TranscriptionWorkflow,
    upload_ttl_seconds: int,
) -> PresignedUpload:
    try:
        return await storage.create_presigned_upload(
            workflow.source_object_key,
            content_type=workflow.source_content_type,
            expires_in_seconds=upload_ttl_seconds,
        )
    except ObjectStorageError as exc:
        raise TranscriptionStorageError("업로드 URL 발급에 실패했습니다.") from exc


async def create_workflow(
    session: AsyncSession,
    storage: ObjectStorage,
    *,
    tenant_id: str,
    idempotency_key: str | None,
    request: TranscriptionCreateRequest,
    upload_ttl_seconds: int,
) -> tuple[WorkflowCreation, bool]:
    if re.fullmatch(r"[0-9A-Za-z_-]+", tenant_id) is None:
        raise OnRampError("유효하지 않은 tenant 식별자입니다.", status_code=400)
    normalized_key = (idempotency_key or "").strip() or None
    if normalized_key:
        existing = await session.scalar(
            select(TranscriptionWorkflow).where(
                TranscriptionWorkflow.tenant_id == tenant_id,
                TranscriptionWorkflow.idempotency_key == normalized_key,
            )
        )
        if existing is not None:
            upload = (
                await _presign(storage, existing, upload_ttl_seconds)
                if existing.status == WorkflowStatus.awaiting_upload
                else None
            )
            return WorkflowCreation(existing, upload), False

    transcription_id = uuid.uuid4()
    workflow = TranscriptionWorkflow(
        transcription_id=transcription_id,
        tenant_id=tenant_id,
        idempotency_key=normalized_key,
        status=WorkflowStatus.awaiting_upload,
        source_object_key=_object_key(tenant_id, transcription_id, request.filename),
        source_filename=_source_filename(request.filename),
        source_content_type=_normalize_content_type(request.content_type),
        source_size_bytes=request.size_bytes,
        title=request.title.strip() or _safe_filename(request.filename),
        language=request.language,
        category=request.category,
    )
    session.add(workflow)
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        if not normalized_key:
            raise
        existing = await session.scalar(
            select(TranscriptionWorkflow).where(
                TranscriptionWorkflow.tenant_id == tenant_id,
                TranscriptionWorkflow.idempotency_key == normalized_key,
            )
        )
        if existing is None:
            raise
        upload = (
            await _presign(storage, existing, upload_ttl_seconds)
            if existing.status == WorkflowStatus.awaiting_upload
            else None
        )
        return WorkflowCreation(existing, upload), False

    upload = await _presign(storage, workflow, upload_ttl_seconds)
    return WorkflowCreation(workflow, upload), True


async def get_workflow(
    session: AsyncSession,
    *,
    tenant_id: str,
    transcription_id: UUID,
    for_update: bool = False,
) -> TranscriptionWorkflow:
    statement = select(TranscriptionWorkflow).where(
        TranscriptionWorkflow.tenant_id == tenant_id,
        TranscriptionWorkflow.transcription_id == transcription_id,
    )
    if for_update:
        statement = statement.with_for_update().execution_options(populate_existing=True)
    workflow = await session.scalar(statement)
    if workflow is None:
        raise TranscriptionNotFoundError
    return workflow


async def complete_upload(
    session: AsyncSession,
    storage: ObjectStorage,
    *,
    tenant_id: str,
    transcription_id: UUID,
    request: UploadCompleteRequest,
) -> TranscriptionWorkflow:
    workflow = await get_workflow(
        session,
        tenant_id=tenant_id,
        transcription_id=transcription_id,
    )
    if workflow.status == WorkflowStatus.queued:
        return workflow
    if workflow.status != WorkflowStatus.awaiting_upload:
        raise TranscriptionConflictError(f"현재 상태에서는 업로드를 완료할 수 없습니다: {workflow.status}")

    try:
        metadata = await storage.head(workflow.source_object_key)
    except ObjectNotFoundError as exc:
        raise TranscriptionConflictError("업로드된 음성 파일을 찾을 수 없습니다.") from exc
    except ObjectStorageError as exc:
        raise TranscriptionStorageError("업로드 파일 검증에 실패했습니다.") from exc

    if metadata.object_key != workflow.source_object_key:
        raise TranscriptionConflictError("업로드 object key가 workflow와 일치하지 않습니다.")
    if metadata.size_bytes != workflow.source_size_bytes or metadata.size_bytes != request.size_bytes:
        raise TranscriptionConflictError("업로드 파일 크기가 요청 정보와 일치하지 않습니다.")
    if _normalize_content_type(metadata.content_type) != _normalize_content_type(workflow.source_content_type):
        raise TranscriptionConflictError("업로드 파일 Content-Type이 요청 정보와 일치하지 않습니다.")
    if _normalize_etag(metadata.etag) != _normalize_etag(request.etag):
        raise TranscriptionConflictError("업로드 파일 ETag가 요청 정보와 일치하지 않습니다.")

    workflow = await get_workflow(
        session,
        tenant_id=tenant_id,
        transcription_id=transcription_id,
        for_update=True,
    )
    if workflow.status == WorkflowStatus.queued:
        return workflow
    if workflow.status != WorkflowStatus.awaiting_upload:
        raise TranscriptionConflictError(f"현재 상태에서는 업로드를 완료할 수 없습니다: {workflow.status}")

    requested_at = utcnow()
    workflow.status = WorkflowStatus.queued
    workflow.source_etag = metadata.etag
    workflow.updated_at = requested_at
    event_id = f"evt_{uuid.uuid4().hex}"
    session.add(
        EventOutbox(
            id=event_id,
            aggregate_type="transcription",
            aggregate_id=str(workflow.transcription_id),
            event_type=TRANSCRIPTION_REQUESTED_EVENT,
            stream_name=STT_REQUEST_STREAM,
            payload_json={
                "schema_version": "1.0",
                "transcription_id": str(workflow.transcription_id),
                "tenant_id": workflow.tenant_id,
                "source_object_key": workflow.source_object_key,
                "source_etag": workflow.source_etag,
                "source_filename": workflow.source_filename,
                "source_content_type": workflow.source_content_type,
                "source_size_bytes": workflow.source_size_bytes,
                "title": workflow.title,
                "language": workflow.language,
                "requested_at": requested_at.isoformat(),
            },
        )
    )
    await session.flush()
    return workflow


def create_response(creation: WorkflowCreation) -> TranscriptionCreateResponse:
    workflow = creation.workflow
    return TranscriptionCreateResponse(
        workflow_id=workflow.id,
        transcription_id=workflow.transcription_id,
        status=workflow.status,
        upload=(
            UploadInstruction(
                method="PUT",
                url=creation.upload.url,
                headers=creation.upload.headers,
                expires_at=creation.upload.expires_at,
            )
            if creation.upload is not None
            else None
        ),
    )


def status_response(workflow: TranscriptionWorkflow) -> TranscriptionStatusResponse:
    processed = workflow.completed_chunks + workflow.failed_chunks
    percent = round((processed / workflow.total_chunks) * 100, 2) if workflow.total_chunks else 0.0
    report_status = "not_started"
    if workflow.status in {WorkflowStatus.report_queued, WorkflowStatus.report_processing}:
        report_status = workflow.status.removeprefix("report_")
    elif workflow.status in {WorkflowStatus.draft, WorkflowStatus.published, WorkflowStatus.report_failed}:
        report_status = workflow.status.value
    return TranscriptionStatusResponse(
        transcription_id=workflow.transcription_id,
        status=workflow.status,
        progress=TranscriptionProgress(
            total_chunks=workflow.total_chunks,
            completed_chunks=workflow.completed_chunks,
            failed_chunks=workflow.failed_chunks,
            percent=percent,
        ),
        report=ReportStatus(status=report_status, report_id=workflow.report_id),
        updated_at=workflow.updated_at,
    )
