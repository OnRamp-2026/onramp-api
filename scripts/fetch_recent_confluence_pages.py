"""Fetch recently modified Confluence pages and write cleaned Markdown files."""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
from pathlib import Path

from app.services.ingest_service import CleanedConfluencePage, IngestService

logger = logging.getLogger(__name__)


def _safe_filename(page: CleanedConfluencePage) -> str:
    title = re.sub(r"[^A-Za-z0-9가-힣._-]+", "-", page.title).strip("-") or "page"
    return f"{page.page_id}-{title}.md"


async def run(hours: int, limit: int, output_dir: str | None, save_html: bool, all_pages: bool = False) -> None:
    service = IngestService()
    pages = await (
        service.clean_all_pages(limit=limit) if all_pages else service.clean_recent_pages(hours=hours, limit=limit)
    )

    if output_dir:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        for page in pages:
            markdown_path = root / _safe_filename(page)
            markdown_path.write_text(page.markdown, encoding="utf-8")
            logger.info("Wrote %s", markdown_path)
            if save_html:
                html_path = markdown_path.with_suffix(".html")
                html_path.write_text(page.html, encoding="utf-8")
                logger.info("Wrote %s", html_path)
        logger.info("Cleaned %s Confluence pages", len(pages))
        return

    for page in pages:
        print(f"# {page.title}")
        print(f"<!-- page_id={page.page_id} last_modified={page.last_modified} -->")
        print()
        print(page.markdown)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confluence 페이지를 정제(Markdown). 증분(--hours) 또는 초기 전체 적재(--all) 모드를 지원한다."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--hours", type=int, help="증분: 최근 N시간 내 수정된 페이지만 (미지정 시 24)")
    mode.add_argument(
        "--all", action="store_true", dest="all_pages", help="전체: 스페이스 전체를 적재(증분 lastmodified 무시)"
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output-dir")
    parser.add_argument("--save-html", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    hours = 24 if args.hours is None else args.hours

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    asyncio.run(
        run(
            hours=hours,
            limit=args.limit,
            output_dir=args.output_dir,
            save_html=args.save_html,
            all_pages=args.all_pages,
        )
    )


if __name__ == "__main__":
    main()
