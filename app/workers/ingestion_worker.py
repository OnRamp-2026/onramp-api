from __future__ import annotations

import asyncio
import logging

from app.config import get_settings
from app.db.models import IndexRun, IndexRunType
from app.db.postgres import session_scope
from app.services import rag_index_repository as repo
from app.services.index_service import IndexProgress, IndexService

logger = logging.getLogger(__name__)


async def _save_progress(run_id, values: IndexProgress) -> None:
    async with session_scope() as db:
        run = await db.get(IndexRun, run_id)
        if run is None:
            return
        await repo.update_index_run_progress(db, run, **values)


async def process_next() -> bool:
    async with session_scope() as db:
        run = await repo.claim_next_index_run(db)
    if run is None:
        return False

    settings = get_settings()
    if run.tenant_id != settings.auth_default_tenant:
        async with session_scope() as db:
            current = await db.get(IndexRun, run.run_id)
            if current is not None:
                await repo.fail_index_run(
                    db,
                    current,
                    error=f"run tenant {run.tenant_id} does not match deployment tenant {settings.auth_default_tenant}",
                )
        return True
    service = IndexService(settings=settings)

    async def progress(values: IndexProgress) -> None:
        await _save_progress(run.run_id, values)

    try:
        if run.run_type == IndexRunType.full_scan.value:
            await service.index_all_pages(limit=1000, run_id=run.run_id, progress=progress)
        else:
            await service.index_recent_pages(hours=24, limit=1000, run_id=run.run_id, progress=progress)
    except Exception as exc:
        logger.exception("ingestion run failed: %s", run.run_id)
        async with session_scope() as db:
            current = await db.get(IndexRun, run.run_id)
            if current is not None:
                await repo.fail_index_run(db, current, error=str(exc))
    return True


async def run_forever() -> None:
    while True:
        processed = await process_next()
        if not processed:
            await asyncio.sleep(2)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_forever())


if __name__ == "__main__":
    main()
