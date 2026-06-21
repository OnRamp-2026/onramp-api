from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import EventOutbox, Report, ReportStatus, TranscriptionWorkflow, WorkflowStatus, utcnow
from app.middleware.error_handler import OnRampError
from app.models.asset_history import AssetDeletionResponse
from app.queue.constants import DELETE_REQUESTED_EVENT_TYPE, STT_REQUEST_STREAM
from app.queue.events import TranscriptionDeleteRequested


async def request_asset_deletion(
    session: AsyncSession,
    *,
    tenant_id: str,
    user_id: str,
    transcription_id: UUID,
) -> AssetDeletionResponse:
    workflow = await session.scalar(
        select(TranscriptionWorkflow)
        .where(
            TranscriptionWorkflow.tenant_id == tenant_id,
            TranscriptionWorkflow.created_by_user_id == user_id,
            TranscriptionWorkflow.transcription_id == transcription_id,
        )
        .with_for_update()
    )
    if workflow is None:
        raise OnRampError("자산화 기록을 찾을 수 없습니다", status_code=404)
    if workflow.status == WorkflowStatus.deleting:
        return AssetDeletionResponse(transcription_id=str(transcription_id), status="deleting")
    if workflow.status != WorkflowStatus.draft:
        raise OnRampError("초안 상태의 자산만 삭제할 수 있습니다", status_code=409)

    report = await session.scalar(
        select(Report)
        .where(
            Report.tenant_id == tenant_id,
            Report.source_transcription_id == transcription_id,
        )
        .with_for_update()
    )
    if report is None or report.status != ReportStatus.draft:
        raise OnRampError("초안 상태의 자산만 삭제할 수 있습니다", status_code=409)

    workflow.status = WorkflowStatus.deleting
    workflow.updated_at = utcnow()
    payload = TranscriptionDeleteRequested(
        transcription_id=transcription_id,
        tenant_id=tenant_id,
    )
    session.add(
        EventOutbox(
            id=f"evt_delete_{transcription_id.hex}",
            aggregate_type="transcription",
            aggregate_id=str(transcription_id),
            event_type=DELETE_REQUESTED_EVENT_TYPE,
            stream_name=STT_REQUEST_STREAM,
            payload_json=payload.model_dump(mode="json"),
        )
    )
    await session.flush()
    return AssetDeletionResponse(transcription_id=str(transcription_id), status="deleting")
