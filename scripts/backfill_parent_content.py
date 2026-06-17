"""chunk_registry.parent_content 백필 (#212 Phase 0-A).

기존 색인분의 parent_content가 전부 NULL이라, parent expansion 측정 전에 채운다.
**임베딩·Qdrant는 건드리지 않는다** — Postgres의 cleaned_markdown을 적재와 동일한 경로
(mask → profile 분류 → chunk)로 재청킹해 parent 본문을 얻고, chunk_registry를 parent_id
기준으로 UPDATE만 한다. 청킹은 결정적이라 parent_id가 기존 행과 일치한다.

전체 재적재(REINDEX=1)와 달리 LLM·임베딩 호출이 없어 비용이 거의 들지 않는다.

사용: DATABASE_URL=... python scripts/backfill_parent_content.py [--dry-run] [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import select, update  # noqa: E402

from app.db.models import ChunkRegistry, SourceDocument  # noqa: E402
from app.db.postgres import session_scope  # noqa: E402
from app.services.ingest_service import CleanedConfluencePage, IngestService  # noqa: E402

logger = logging.getLogger(__name__)


def _parents_for(ingest: IngestService, doc: SourceDocument) -> dict[str, str]:
    """적재와 동일한 경로(mask→profile→chunk)로 재청킹해 parent_id→content 맵을 만든다."""
    cleaned = CleanedConfluencePage(
        page_id=doc.page_id,
        title=doc.title or "",
        space_key=doc.space_key or "",
        markdown=doc.cleaned_markdown or "",
        html="",
        last_modified="",
        version=None,
        url=doc.source_url or "",
    )
    masked = ingest._mask_page(cleaned)
    profile = ingest.profile_classifier.classify_page(masked.title, masked.markdown)
    chunked = ingest._chunk_cleaned_page(masked, chunking_profile=profile)
    return {p.parent_id: p.content for p in chunked.parents}


async def run(*, dry_run: bool, limit: int | None) -> None:
    ingest = IngestService()
    docs_done = 0
    parents_updated = 0
    rows_updated = 0
    async with session_scope() as db:
        stmt = select(SourceDocument).where(SourceDocument.cleaned_markdown.isnot(None))
        if limit:
            stmt = stmt.limit(limit)
        docs = (await db.scalars(stmt)).all()
        logger.info("대상 문서 %d개", len(docs))
        for doc in docs:
            pmap = _parents_for(ingest, doc)
            if not pmap:
                continue
            for parent_id, content in pmap.items():
                if dry_run:
                    parents_updated += 1
                    continue
                result = await db.execute(
                    update(ChunkRegistry)
                    .where(
                        ChunkRegistry.tenant_id == doc.tenant_id,
                        ChunkRegistry.source == doc.source,
                        ChunkRegistry.parent_id == parent_id,
                    )
                    .values(parent_content=content)
                )
                rows_updated += getattr(result, "rowcount", 0) or 0
                parents_updated += 1
            docs_done += 1
            if docs_done % 100 == 0:
                logger.info("진행 %d/%d 문서", docs_done, len(docs))
        if dry_run:
            logger.info("[dry-run] 문서 %d → parent %d개 채울 예정 (UPDATE 미실행)", len(docs), parents_updated)
        else:
            await db.commit()  # session_scope는 auto-commit 안 함
            logger.info(
                "완료: 문서 %d, parent %d개, chunk_registry 행 %d개 갱신", docs_done, parents_updated, rows_updated
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="chunk_registry.parent_content 백필 (#212).")
    parser.add_argument("--dry-run", action="store_true", help="UPDATE 없이 채울 parent 수만 계산")
    parser.add_argument("--limit", type=int, default=None, help="처리할 문서 수(검증용)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    asyncio.run(run(dry_run=args.dry_run, limit=args.limit))


if __name__ == "__main__":
    main()
