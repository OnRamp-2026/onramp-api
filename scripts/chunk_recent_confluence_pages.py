"""Fetch recent Confluence pages, clean them, and write chunk JSONL files."""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.rag.chunker import child_chunk_to_index_record, write_jsonl  # noqa: E402
from app.services.ingest_service import ChunkedConfluencePage, IngestService  # noqa: E402

logger = logging.getLogger(__name__)


def _safe_stem(page: ChunkedConfluencePage) -> str:
    title = re.sub(r"[^A-Za-z0-9가-힣._-]+", "-", page.page.title).strip("-") or "page"
    return f"{page.page.page_id}-{title}"


async def run(hours: int, limit: int, output_dir: str, parents_output_dir: str | None, all_pages: bool = False) -> None:
    """Run the local batch entrypoint for recent-page (or full) chunking."""

    service = IngestService()
    pages = await (
        service.chunk_all_pages(limit=limit) if all_pages else service.chunk_recent_pages(hours=hours, limit=limit)
    )
    chunks_root = Path(output_dir)
    parents_root = Path(parents_output_dir) if parents_output_dir else None

    for page in pages:
        stem = _safe_stem(page)
        chunk_records = [child_chunk_to_index_record(child) for child in page.children]
        chunks_path = chunks_root / f"{stem}.jsonl"
        write_jsonl(chunks_path, chunk_records)
        logger.info("Wrote %s child chunks to %s", len(page.children), chunks_path)

        if parents_root:
            parents_path = parents_root / f"{stem}.jsonl"
            write_jsonl(parents_path, page.parents)
            logger.info("Wrote %s parent chunks to %s", len(page.parents), parents_path)

    logger.info("Chunked %s Confluence pages", len(pages))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch recent Confluence pages and write semantic chunk JSONL.")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window for Confluence modified pages.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of Confluence pages to fetch.")
    parser.add_argument(
        "--all", action="store_true", dest="all_pages", help="전체 스페이스 적재(증분 lastmodified 무시)"
    )
    parser.add_argument("--output-dir", default="data/processed/chunks", help="Directory for child chunk JSONL files.")
    parser.add_argument("--parents-output-dir", help="Optional directory for parent chunk JSONL files.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    asyncio.run(
        run(
            hours=args.hours,
            limit=args.limit,
            output_dir=args.output_dir,
            parents_output_dir=args.parents_output_dir,
            all_pages=args.all_pages,
        )
    )


if __name__ == "__main__":
    main()
