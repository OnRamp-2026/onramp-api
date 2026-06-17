from __future__ import annotations

import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import PurePath
from uuid import UUID

import httpx
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import TranscriptionWorkflow, WorkflowStatus, utcnow
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
from app.services.stt_result_client import SttCreateTranscriptionResponse, SttResultClient
from app.storage.base import PresignedUpload


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


def _normalize_content_type(value: str) -> str:
    return value.split(";", 1)[0].strip().lower()


def _stt_upload_to_presigned(stt: SttCreateTranscriptionResponse) -> PresignedUpload:
    return PresignedUpload(
        method=stt.upload.method,
        url=stt.upload.url,
        headers=stt.upload.headers,
        expires_at=stt.upload.expires_at,
    )


async def _call_stt_create(
    stt_client: SttResultClient,
    workflow: TranscriptionWorkflow,
    idempotency_key: str | None,
) -> PresignedUpload:
    try:
        result = await stt_client.create_transcription(
            tenant_id=workflow.tenant_id,
            transcription_id=workflow.transcription_id,
            filename=workflow.source_filename,
            content_type=workflow.source_content_type,
            size_bytes=workflow.source_size_bytes,
            idempotency_key=idempotency_key,
        )
    except httpx.HTTPStatusError as exc:
        raise TranscriptionStorageError("STT API 업로드 URL 발급에 실패했습니다.") from exc
    workflow.source_object_key = result.source_object_key
    return _stt_upload_to_presigned(result)


async def create_workflow(
    session: AsyncSession,
    stt_client: SttResultClient,
    *,
    tenant_id: str,
    idempotency_key: str | None,
    request: TranscriptionCreateRequest,
) -> tuple[WorkflowCreation, bool]:
    if re.fullmatch(r"[0-9A-Za-z_-]+", tenant_id) is None:
        raise OnRampError("유효하지 않은 tenant 식별자입니다.", status_code=400)
    normalized_key = idempotency_key.strip() if idempotency_key is not None else None
    if idempotency_key is not None and not normalized_key:
        raise OnRampError("Idempotency-Key는 공백일 수 없습니다.", status_code=400)
    if normalized_key:
        existing = await session.scalar(
            select(TranscriptionWorkflow).where(
                TranscriptionWorkflow.tenant_id == tenant_id,
                TranscriptionWorkflow.idempotency_key == normalized_key,
            )
        )
        if existing is not None:
            upload = (
                await _call_stt_create(stt_client, existing, normalized_key)
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
        source_object_key="",  # will be set by STT API response
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
            await _call_stt_create(stt_client, existing, normalized_key)
            if existing.status == WorkflowStatus.awaiting_upload
            else None
        )
        return WorkflowCreation(existing, upload), False

    upload = await _call_stt_create(stt_client, workflow, normalized_key)
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
    stt_client: SttResultClient,
    *,
    tenant_id: str,
    transcription_id: UUID,
    request: UploadCompleteRequest,
) -> TranscriptionWorkflow:
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

    try:
        await stt_client.complete_upload(
            transcription_id,
            etag=request.etag,
            size_bytes=request.size_bytes,
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 409:
            raise TranscriptionConflictError("업로드 검증에 실패했습니다.") from exc
        raise TranscriptionStorageError("STT API 업로드 완료 처리에 실패했습니다.") from exc

    workflow.status = WorkflowStatus.queued
    workflow.source_etag = request.etag
    workflow.updated_at = utcnow()
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
    if workflow.status in {
        WorkflowStatus.report_queued,
        WorkflowStatus.report_processing,
        WorkflowStatus.report_failed,
    }:
        report_status = workflow.status.value.removeprefix("report_")
    elif workflow.status in {WorkflowStatus.draft, WorkflowStatus.published}:
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
