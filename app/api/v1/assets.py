from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import AssetUser, DatabaseSession
from app.models.asset_history import AssetHistoryListResponse, AssetHistoryStatus
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
