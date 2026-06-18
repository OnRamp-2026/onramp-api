from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings  # noqa: E402
from app.db.models import IndexRunTrigger  # noqa: E402
from app.db.postgres import session_scope  # noqa: E402
from app.services.ingestion_run_service import enqueue_run  # noqa: E402

logger = logging.getLogger(__name__)


async def run(mode: str, trigger: str) -> int:
    settings = get_settings()
    async with session_scope() as db:
        queued = await enqueue_run(db, tenant_id=settings.auth_default_tenant, mode=mode, trigger=trigger)
    if queued is None:
        logger.info("active ingestion run already exists")
        return 0
    logger.info("queued ingestion run %s", queued.run_id)
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("incremental", "full_scan"), default="incremental")
    parser.add_argument("--trigger", choices=("cron", "manual"), default=IndexRunTrigger.cron.value)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.mode, args.trigger)))


if __name__ == "__main__":
    main()
