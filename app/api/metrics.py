from __future__ import annotations

from fastapi import APIRouter, Response

from app.api.deps import DatabaseSession
from app.db.redis import get_redis
from app.services.prometheus_metrics import collect_worker_metric_snapshot, render_worker_metrics

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
async def metrics(session: DatabaseSession) -> Response:
    snapshot = await collect_worker_metric_snapshot(session, get_redis())
    return Response(
        content=render_worker_metrics(snapshot),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
