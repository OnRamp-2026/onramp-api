"""PostgreSQL 원장(source_document)의 원문을 OpenSearch 문서 인덱스(onramp-documents)에 투영.

청크 BM25(onramp-chunks)와 별개로, **문서 전체 원문**을 색인해 document_tools(전문 조회·문서 단위
BM25)에 사용한다. 원문 진실원천은 Postgres이고, OpenSearch는 검색용 파생 인덱스다.

예) python scripts/index_documents_to_opensearch.py            # 기본 테넌트 전체
    python scripts/index_documents_to_opensearch.py --tenant onramp --source github
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from sqlalchemy import select

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings  # noqa: E402
from app.db.opensearch import close_opensearch, get_opensearch  # noqa: E402
from app.db.postgres import session_scope  # noqa: E402
from app.db.models import SourceDocument  # noqa: E402

logger = logging.getLogger(__name__)


async def run(tenant_id: str, source: str | None) -> None:
    async with session_scope() as db:
        stmt = select(SourceDocument).where(SourceDocument.tenant_id == tenant_id)
        if source:
            stmt = stmt.where(SourceDocument.source == source)
        rows = (await db.execute(stmt)).scalars().all()

    documents = [
        {
            "tenant_id": row.tenant_id,
            "doc_id": row.page_id,
            "source": row.source,
            "title": row.title,
            "domain": row.domain or "",
            "content": row.cleaned_markdown or "",
        }
        for row in rows
        if (row.cleaned_markdown or "").strip()
    ]

    skipped = len(rows) - len(documents)
    if not documents:
        logger.warning("투영할 문서 없음 (tenant=%s, source=%s, 원문 없는 행 %d개 스킵)", tenant_id, source, skipped)
        return

    client = get_opensearch()
    try:
        await client.upsert_documents(documents)
    finally:
        await close_opensearch()

    logger.info(
        "OpenSearch 문서 인덱스 투영 완료: %d개 (스킵 %d, tenant=%s, source=%s)",
        len(documents),
        skipped,
        tenant_id,
        source or "all",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Postgres 원장 원문을 OpenSearch 문서 인덱스로 투영.")
    parser.add_argument("--tenant", default=None, help="테넌트(기본: auth_default_tenant)")
    parser.add_argument("--source", default=None, help="필터: confluence | github (미지정 시 전체)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    tenant_id = args.tenant or get_settings().auth_default_tenant
    asyncio.run(run(tenant_id, args.source))


if __name__ == "__main__":
    main()
