from __future__ import annotations

import asyncio

import structlog

from app.config import get_settings
from app.db.postgres import get_session_factory
from app.services.report_worker import ReportWorker
from app.services.stt_result_client import SttResultClient

logger = structlog.get_logger(__name__)


async def run() -> None:
    settings = get_settings()
    worker = ReportWorker(
        get_session_factory(),
        SttResultClient(
            settings.stt_service_base_url,
            settings.stt_service_token.get_secret_value(),
            settings.stt_result_timeout_seconds,
        ),
    )
    poll_interval = settings.report_worker_poll_interval_ms / 1000
    while True:
        try:
            processed = await worker.process_next()
        except Exception:
            await logger.aexception("report_worker_loop_failed")
            await asyncio.sleep(poll_interval)
            continue
        if not processed:
            await asyncio.sleep(poll_interval)


if __name__ == "__main__":
    asyncio.run(run())
