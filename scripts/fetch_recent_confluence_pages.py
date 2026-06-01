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


async def run(hours: int, limit: int, output_dir: str | None, save_html: bool) -> None:
    pages = await IngestService().clean_recent_pages(hours=hours, limit=limit)

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
    parser = argparse.ArgumentParser(description="Fetch recently modified Confluence pages and clean them.")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output-dir")
    parser.add_argument("--save-html", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    asyncio.run(run(hours=args.hours, limit=args.limit, output_dir=args.output_dir, save_html=args.save_html))


if __name__ == "__main__":
    main()
