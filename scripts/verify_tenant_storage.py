from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import httpx
from sqlalchemy import func, select, text

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings  # noqa: E402
from app.db.models import SourceDocument  # noqa: E402
from app.db.postgres import session_scope  # noqa: E402

logger = logging.getLogger(__name__)


async def _postgres_counts(expected_tenant: str) -> tuple[int, int]:
    async with session_scope() as db:
        total = int((await db.scalar(select(func.count()).select_from(SourceDocument))) or 0)
        outside = int(
            (
                await db.scalar(
                    text(
                        """
                        SELECT COUNT(*) FROM (
                            SELECT tenant_id FROM transcription_workflows
                            UNION ALL SELECT tenant_id FROM report_jobs
                            UNION ALL SELECT tenant_id FROM reports
                            UNION ALL SELECT tenant_id FROM source_document
                            UNION ALL SELECT tenant_id FROM source_document_previous
                            UNION ALL SELECT tenant_id FROM index_run
                            UNION ALL SELECT tenant_id FROM chunk_registry
                            UNION ALL SELECT tenant_id FROM chat_log
                            UNION ALL SELECT tenant_id FROM conversation
                            UNION ALL SELECT tenant_id FROM message
                        ) tenant_rows
                        WHERE tenant_id <> :expected_tenant
                        """
                    ),
                    {"expected_tenant": expected_tenant},
                )
            )
            or 0
        )
    return total, outside


async def _http_counts(expected_tenant: str) -> list[tuple[str, int, int]]:
    s = get_settings()
    async with httpx.AsyncClient(timeout=15) as client:
        qdrant_base = f"http://{s.qdrant_host}:{s.qdrant_port}"
        total_response = await client.post(
            f"{qdrant_base}/collections/{s.qdrant_collection}/points/count",
            json={"exact": True},
        )
        total_response.raise_for_status()
        tenant_response = await client.post(
            f"{qdrant_base}/collections/{s.qdrant_collection}/points/count",
            json={
                "exact": True,
                "filter": {"must": [{"key": "tenant_id", "match": {"value": expected_tenant}}]},
            },
        )
        tenant_response.raise_for_status()
        counts = [
            (
                "qdrant",
                int(total_response.json()["result"]["count"]),
                int(tenant_response.json()["result"]["count"]),
            )
        ]

        os_base = f"{s.opensearch_scheme}://{s.opensearch_host}:{s.opensearch_port}"
        for index in (s.opensearch_index, s.opensearch_documents_index):
            total = await client.get(f"{os_base}/{index}/_count")
            total.raise_for_status()
            tenant = await client.post(
                f"{os_base}/{index}/_count",
                json={"query": {"term": {"tenant_id": expected_tenant}}},
            )
            tenant.raise_for_status()
            counts.append((index, int(total.json()["count"]), int(tenant.json()["count"])))
    return counts


async def run(expected_tenant: str) -> int:
    postgres_total, postgres_outside = await _postgres_counts(expected_tenant)
    logger.info("postgres documents=%d outside_tenant=%d", postgres_total, postgres_outside)
    failures = int(postgres_outside != 0)

    for store, total, tenant_count in await _http_counts(expected_tenant):
        logger.info("%s total=%d expected_tenant=%d", store, total, tenant_count)
        failures += int(total != tenant_count)
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-tenant", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(asyncio.run(run(args.expected_tenant)))


if __name__ == "__main__":
    main()
