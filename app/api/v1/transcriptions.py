from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header, Response, status

from app.api.deps import CurrentTenant, DatabaseSession, SttClientDependency
from app.models.transcription import (
    TranscriptionCreateRequest,
    TranscriptionCreateResponse,
    TranscriptionStatusResponse,
    UploadCompleteRequest,
    UploadCompleteResponse,
)
from app.services.transcription_service import (
    complete_upload,
    create_response,
    create_workflow,
    get_workflow,
    status_response,
)

router = APIRouter(prefix="/transcriptions")


@router.post("", response_model=TranscriptionCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_transcription(
    request: TranscriptionCreateRequest,
    response: Response,
    session: DatabaseSession,
    stt_client: SttClientDependency,
    tenant_id: CurrentTenant,
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key", max_length=255),
    ] = None,
) -> TranscriptionCreateResponse:
    creation, created = await create_workflow(
        session,
        stt_client,
        tenant_id=tenant_id,
        idempotency_key=idempotency_key,
        request=request,
    )
    await session.commit()
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return create_response(creation)


@router.post(
    "/{transcription_id}/upload-complete",
    response_model=UploadCompleteResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def mark_upload_complete(
    transcription_id: UUID,
    request: UploadCompleteRequest,
    session: DatabaseSession,
    stt_client: SttClientDependency,
    tenant_id: CurrentTenant,
) -> UploadCompleteResponse:
    workflow = await complete_upload(
        session,
        stt_client,
        tenant_id=tenant_id,
        transcription_id=transcription_id,
        request=request,
    )
    await session.commit()
    return UploadCompleteResponse(
        transcription_id=workflow.transcription_id,
        status=workflow.status,
    )


@router.get("/{transcription_id}", response_model=TranscriptionStatusResponse)
async def get_transcription_status(
    transcription_id: UUID,
    session: DatabaseSession,
    tenant_id: CurrentTenant,
) -> TranscriptionStatusResponse:
    workflow = await get_workflow(
        session,
        tenant_id=tenant_id,
        transcription_id=transcription_id,
    )
    return status_response(workflow)
