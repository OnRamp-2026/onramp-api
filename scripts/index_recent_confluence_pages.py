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


async def run(hours: int, limit: int) -> None:
    """Run recent Confluence indexing."""

    result = await IndexService().index_recent_pages(hours=hours, limit=limit)
    logger.info("Indexed %s child chunks from %s Confluence pages", result.chunks_indexed, result.pages_indexed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index recent Confluence pages into Qdrant.")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window for Confluence modified pages.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of Confluence pages to fetch.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    asyncio.run(run(hours=args.hours, limit=args.limit))


if __name__ == "__main__":
    main()
