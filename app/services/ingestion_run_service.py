from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import IndexRun, IndexRunTrigger, IndexRunType
from app.services import rag_index_repository as repo


def normalize_run_type(mode: str) -> str:
    if mode == "incremental":
        return IndexRunType.incremental.value
    if mode == "full_scan":
        return IndexRunType.full_scan.value
    raise ValueError(f"unsupported ingestion mode: {mode}")


async def enqueue_run(
    db: AsyncSession,
    *,
    tenant_id: str,
    mode: str,
    trigger: str = IndexRunTrigger.manual.value,
) -> IndexRun | None:
    return await repo.enqueue_index_run(
        db,
        tenant_id=tenant_id,
        run_type=normalize_run_type(mode),
        trigger=trigger,
    )
