from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, status

from app.api.deps import AssetUser, DatabaseSession
from app.models.asset_history import AssetDeletionResponse, AssetHistoryListResponse, AssetHistoryStatus
from app.services.asset_deletion_service import request_asset_deletion
from app.services.asset_history_service import list_assets

router = APIRouter(prefix="/assets")


@router.get("", response_model=AssetHistoryListResponse)
async def get_assets(
    session: DatabaseSession,
    user: AssetUser,
    status: AssetHistoryStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> AssetHistoryListResponse:
    return await list_assets(
        session,
        tenant_id=user.tenant_id,
        user_id=user.subject,
        status=status,
        limit=limit,
    )


@router.delete("/{transcription_id}", response_model=AssetDeletionResponse, status_code=status.HTTP_202_ACCEPTED)
async def delete_asset(
    transcription_id: UUID,
    session: DatabaseSession,
    user: AssetUser,
) -> AssetDeletionResponse:
    response = await request_asset_deletion(
        session,
        tenant_id=user.tenant_id,
        user_id=user.subject,
        transcription_id=transcription_id,
    )
    await session.commit()
    return response
