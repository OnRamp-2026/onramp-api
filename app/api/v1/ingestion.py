from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import CurrentTenant, DatabaseSession
from app.db.models import IndexRun
from app.models.ingestion import (
    IngestionRunCreate,
    IngestionRunListResponse,
    IngestionRunResponse,
)
from app.services import rag_index_repository as repo
from app.services.ingestion_run_service import enqueue_run

router = APIRouter(prefix="/ingestion", tags=["Ingestion"])


def _response(run: IndexRun) -> IngestionRunResponse:
    return IngestionRunResponse(
        run_id=run.run_id,
        tenant_id=run.tenant_id,
        mode=run.run_type,
        trigger=run.trigger,
        status=run.status,
        stage=run.stage,
        pages_discovered=run.pages_discovered,
        pages_processed=run.pages_processed,
        pages_indexed=run.pages_indexed,
        pages_skipped=run.pages_skipped,
        pages_failed=run.pages_failed,
        chunks_indexed=run.chunks_indexed,
        chunks_deleted=run.chunks_deleted,
        started_at=run.started_at,
        finished_at=run.finished_at,
        error_message=run.error_message,
    )


@router.post("/runs", response_model=IngestionRunResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_run(
    body: IngestionRunCreate,
    tenant_id: CurrentTenant,
    db: DatabaseSession,
) -> IngestionRunResponse:
    run = await enqueue_run(db, tenant_id=tenant_id, mode=body.mode)
    if run is None:
        active = await repo.get_active_index_run(db, tenant_id=tenant_id)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "이미 실행 중인 수집 작업이 있습니다.",
                "run_id": str(active.run_id) if active else None,
            },
        )
    return _response(run)


@router.get("/runs/current", response_model=IngestionRunResponse | None)
async def current_run(tenant_id: CurrentTenant, db: DatabaseSession) -> IngestionRunResponse | None:
    run = await repo.get_active_index_run(db, tenant_id=tenant_id)
    return _response(run) if run else None


@router.get("/runs", response_model=IngestionRunListResponse)
async def run_history(
    tenant_id: CurrentTenant,
    db: DatabaseSession,
    limit: int = Query(default=10, ge=1, le=50),
) -> IngestionRunListResponse:
    runs = await repo.list_index_runs(db, tenant_id=tenant_id, limit=limit)
    return IngestionRunListResponse(runs=[_response(run) for run in runs])
