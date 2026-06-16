"""적재 현황 — Postgres(source별 문서·청크), Qdrant 포인트, OpenSearch 청크/문서 카운트.

로컬·prod 무관하게 동작한다(앱 config의 DATABASE_URL/QDRANT_*/OPENSEARCH_* 를 그대로 사용).
docker나 kubectl 없이 앱이 보는 저장소를 직접 조회하므로 어느 환경에서든 같은 결과를 준다.

예) python scripts/ingest_status.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
from sqlalchemy import func, select

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings  # noqa: E402
from app.db.models import ChunkRegistry, SourceDocument  # noqa: E402
from app.db.postgres import session_scope  # noqa: E402


async def main() -> None:
    s = get_settings()

    # PostgreSQL — 원문 원장
    async with session_scope() as db:
        by_source = (
            await db.execute(select(SourceDocument.source, func.count()).group_by(SourceDocument.source))
        ).all()
        chunk_rows = await db.scalar(select(func.count()).select_from(ChunkRegistry))
    print("[PostgreSQL] source_document:")
    for source, n in by_source or [("(없음)", 0)]:
        print(f"    {source}: {n} docs")
    print(f"[PostgreSQL] chunk_registry: {chunk_rows} rows")

    async with httpx.AsyncClient(timeout=10) as client:
        # Qdrant — dense 청크
        try:
            r = await client.get(f"http://{s.qdrant_host}:{s.qdrant_port}/collections/{s.qdrant_collection}")
            if r.status_code == 404:
                print(f"[Qdrant] {s.qdrant_collection}: (컬렉션 없음)")
            else:
                print(f"[Qdrant] {s.qdrant_collection}: {r.json()['result']['points_count']} points")
        except Exception as exc:  # noqa: BLE001
            print(f"[Qdrant] 조회 실패: {exc}")

        # OpenSearch — 청크 BM25 + 문서 BM25
        base = f"{s.opensearch_scheme}://{s.opensearch_host}:{s.opensearch_port}"
        for index in (s.opensearch_index, s.opensearch_documents_index):
            try:
                r = await client.get(f"{base}/{index}/_count")
                if r.status_code == 404:
                    print(f"[OpenSearch] {index}: (인덱스 없음)")
                else:
                    print(f"[OpenSearch] {index}: {r.json().get('count')} docs")
            except Exception as exc:  # noqa: BLE001
                print(f"[OpenSearch] {index} 조회 실패: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
