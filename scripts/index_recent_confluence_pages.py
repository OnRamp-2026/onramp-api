"""Fetch recent Confluence pages, prepare chunks, embed, and upsert to Qdrant."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.index_service import IndexService  # noqa: E402

logger = logging.getLogger(__name__)


async def run(hours: int, limit: int, all_pages: bool = False) -> None:
    """Run recent (or full) Confluence indexing."""

    service = IndexService()
    result = await (
        service.index_all_pages(limit=limit) if all_pages else service.index_recent_pages(hours=hours, limit=limit)
    )
    logger.info("Indexed %s child chunks from %s Confluence pages", result.chunks_indexed, result.pages_indexed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index recent Confluence pages into Qdrant.")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window for Confluence modified pages.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of Confluence pages to fetch.")
    parser.add_argument(
        "--all", action="store_true", dest="all_pages", help="전체 스페이스 적재(증분 lastmodified 무시)"
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    asyncio.run(run(hours=args.hours, limit=args.limit, all_pages=args.all_pages))


if __name__ == "__main__":
    main()
