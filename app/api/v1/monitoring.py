from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.deps import CurrentUser, DatabaseSession
from app.services.monitoring_service import get_detail, get_overview

router = APIRouter(prefix="/monitoring")


@router.get("/overview")
async def monitoring_overview(
    user: CurrentUser,
    db: DatabaseSession,
    scope: str = Query(...),
    period: str = Query(...),
) -> dict:
    return await get_overview(db, user=user, scope=scope, period=period)


@router.get("/details/{metric_id}")
async def monitoring_detail(
    metric_id: str,
    user: CurrentUser,
    db: DatabaseSession,
    scope: str = Query(...),
    period: str = Query(...),
) -> dict:
    return await get_detail(db, user=user, scope=scope, period=period, metric_id=metric_id)
