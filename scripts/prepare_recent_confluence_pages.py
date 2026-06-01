"""Fetch, clean, mask, chunk, classify, and write embedding-ready JSONL files."""

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
    title = title[:120].rstrip("-.") or "page"
    return f"{page.page.page_id}-{title}"


async def run(hours: int, limit: int, output_dir: str, parents_output_dir: str | None) -> None:
    """Run the local batch entrypoint for embedding-ready chunks."""

    service = IngestService()
    pages = await service.prepare_recent_pages_for_embedding(hours=hours, limit=limit)
    chunks_root = Path(output_dir)
    parents_root = Path(parents_output_dir) if parents_output_dir else None

    for page in pages:
        stem = _safe_stem(page)
        chunks_path = chunks_root / f"{stem}.jsonl"
        write_jsonl(chunks_path, [child_chunk_to_index_record(child) for child in page.children])
        logger.info("Wrote %s classified child chunks to %s", len(page.children), chunks_path)

        if parents_root:
            parents_path = parents_root / f"{stem}.jsonl"
            write_jsonl(parents_path, page.parents)
            logger.info("Wrote %s parent chunks to %s", len(page.parents), parents_path)

    logger.info("Prepared %s Confluence pages for embedding", len(pages))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write masked and classified recent Confluence chunk JSONL.")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window for Confluence modified pages.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of Confluence pages to fetch.")
    parser.add_argument("--output-dir", default="data/processed/prepared_chunks")
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
        )
    )


if __name__ == "__main__":
    main()
