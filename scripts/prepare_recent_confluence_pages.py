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


async def run(hours: int, limit: int, output_dir: str, parents_output_dir: str | None, all_pages: bool = False) -> None:
    """Run the local batch entrypoint for embedding-ready chunks."""

    service = IngestService()
    pages = await (
        service.prepare_all_pages_for_embedding(limit=limit)
        if all_pages
        else service.prepare_recent_pages_for_embedding(hours=hours, limit=limit)
    )
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
    parser = argparse.ArgumentParser(
        description="마스킹·분류된 임베딩용 청크 JSONL 생성. 증분(--hours) 또는 초기 전체 적재(--all) 모드를 지원한다."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--hours", type=int, help="증분: 최근 N시간 내 수정된 페이지만 (미지정 시 24)")
    mode.add_argument(
        "--all", action="store_true", dest="all_pages", help="전체: 스페이스 전체를 적재(증분 lastmodified 무시)"
    )
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of Confluence pages to fetch.")
    parser.add_argument("--output-dir", default="data/processed/prepared_chunks")
    parser.add_argument("--parents-output-dir", help="Optional directory for parent chunk JSONL files.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    hours = 24 if args.hours is None else args.hours

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    asyncio.run(
        run(
            hours=hours,
            limit=args.limit,
            output_dir=args.output_dir,
            parents_output_dir=args.parents_output_dir,
            all_pages=args.all_pages,
        )
    )


if __name__ == "__main__":
    main()
