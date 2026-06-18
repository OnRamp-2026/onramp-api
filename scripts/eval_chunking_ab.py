"""청킹 방식 A/B — config-hash 임시 컬렉션 재색인 + page-level 검색 게이트 (#212 step 5).

각 청킹 구성(onramp/token/markdown/recursive)을 `SourceDocument.cleaned_markdown`으로 재청킹해
**임시 Qdrant 컬렉션**(`onramp-eval-<strategy>-<hash>`)에 dense 색인하고, 골든셋으로 **splitter-독립
page-level** 검색 지표(#212 §2-2)를 측정한다.

설계 원칙:
- production 컬렉션을 건드리지 않는다 — 임시 이름 + 끝나면 삭제(--keep로 보존).
- **dense-only 게이트** — BM25/리랭커는 직교 레버라 분리(`bm25_search_enabled=false`, mode=dense).
- **domain 미적용**(domains=None) — 청킹 효과만 격리(도메인 가산은 별도 레버).
- settings 싱글톤 override로 색인·검색 모두 임시 컬렉션을 가리키게 한다(코어 변경 없음).

전제: 라이브 Qdrant + Postgres(cleaned_markdown) + OpenAI 임베딩 (비용 발생).
사용:
    python scripts/eval_chunking_ab.py --strategy token
    python scripts/eval_chunking_ab.py --strategy onramp --doc-limit 50   # 소규모 먼저
    python scripts/eval_chunking_ab.py --strategy recursive --keep        # 임시 컬렉션 보존
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

from sqlalchemy import select  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db.models import SourceDocument  # noqa: E402
from app.db.postgres import session_scope  # noqa: E402
from app.eval.chunking_experiment import ChunkingConfig, chunk_page, page_from_row  # noqa: E402
from app.eval.dataset import load_golden_set  # noqa: E402
from app.eval.metrics import aggregate, collapse_to_pages  # noqa: E402
from app.eval.retrieval_adapter import retrieve_for_eval  # noqa: E402
from app.rag.chunker import ChildChunk  # noqa: E402
from app.rag.indexer import index_children  # noqa: E402

logger = logging.getLogger(__name__)


async def _load_pages(doc_limit: int | None):
    """cleaned_markdown 있는 SourceDocument → MarkdownPage 리스트(코퍼스 스냅샷)."""
    async with session_scope() as db:
        stmt = select(SourceDocument).where(SourceDocument.cleaned_markdown.isnot(None))
        if doc_limit is not None:
            stmt = stmt.limit(doc_limit)
        rows = (await db.execute(stmt)).scalars().all()
    return [
        page_from_row(
            page_id=r.page_id,
            title=r.title,
            markdown=r.cleaned_markdown,
            source_url=r.source_url,
            space_key=r.space_key,
            last_modified=r.last_modified.isoformat() if r.last_modified else "",
        )
        for r in rows
    ]


async def _reindex(config: ChunkingConfig, doc_limit: int | None) -> int:
    """코퍼스를 config 전략으로 재청킹해 임시 컬렉션(=override된 settings.qdrant_collection)에 색인."""
    pages = await _load_pages(doc_limit)
    children: list[ChildChunk] = []
    for page in pages:
        children.extend(chunk_page(config, page))
    logger.info("재청킹: %d pages → %d child chunks (전략=%s)", len(pages), len(children), config.strategy)
    if not children:
        raise RuntimeError("청크가 0개 — cleaned_markdown 코퍼스가 비었는지 확인하세요")
    await index_children(children)  # settings.qdrant_collection(=임시) 사용
    return len(children)


async def _eval_page_level(golden, *, top_k: int, top_n: int):
    """임시 컬렉션을 dense로 검색해 page-level 매크로 평균을 반환(domain 미적용 — 청킹 격리)."""
    page_per_query: list[tuple[list[str], set[str]]] = []
    for g in golden:
        result = await retrieve_for_eval(g.query, mode="dense", domains=None, top_k=top_k, top_n=top_n)
        page_per_query.append((collapse_to_pages(result.chunk_ids), set(g.page_ids)))
    return aggregate(page_per_query)


async def run(args) -> int:
    config = ChunkingConfig(strategy=args.strategy, chunk_tokens=args.chunk_tokens, chunk_overlap=args.chunk_overlap)
    settings = get_settings()
    collection = config.collection_name(args.collection_prefix)
    top_k = args.top_k if args.top_k is not None else settings.retriever_top_k
    top_n = args.top_n if args.top_n is not None else settings.retriever_top_n

    # 임시 컬렉션 + dense-only로 override (색인·검색이 모두 이 settings를 호출 시점에 읽는다).
    settings.qdrant_collection = collection
    settings.bm25_search_enabled = False
    logger.info("임시 컬렉션=%s · config_hash=%s · dense-only", collection, config.hash)

    n_chunks = await _reindex(config, args.doc_limit)

    golden = load_golden_set(args.queries, args.qrels)
    summary = await _eval_page_level(golden, top_k=top_k, top_n=top_n)

    print(f"\n=== 청킹 A/B (page-level, dense, domain 미적용) — {args.strategy} ===")
    print(f"config_hash={config.hash}  collection={collection}  chunks={n_chunks}")
    for key, val in summary.as_dict().items():
        print(f"  {key:<14}: {val}")
    print(f"  n(page_ids 보유 평가질문) = {summary.n}")

    if not args.keep:
        from app.db.qdrant import get_qdrant

        get_qdrant().delete_collection(collection_name=collection)
        logger.info("임시 컬렉션 삭제: %s (보존하려면 --keep)", collection)
    else:
        logger.info("임시 컬렉션 보존: %s", collection)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="청킹 방식 A/B — 임시 컬렉션 재색인 + page-level 게이트 (#212).")
    parser.add_argument("--strategy", required=True, choices=["onramp", "token", "markdown", "recursive"])
    parser.add_argument("--chunk-tokens", type=int, default=400, help="비교군 splitter 청크 토큰 크기(onramp 무관)")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="비교군 splitter 오버랩(onramp 무관)")
    parser.add_argument("--doc-limit", type=int, default=None, help="재색인 문서 수 제한(소규모 먼저)")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--collection-prefix", default="onramp-eval")
    parser.add_argument("--keep", action="store_true", help="끝나도 임시 컬렉션을 삭제하지 않음")
    parser.add_argument("--queries", type=Path, default=ROOT_DIR / "data" / "eval" / "queries.jsonl")
    parser.add_argument("--qrels", type=Path, default=ROOT_DIR / "data" / "eval" / "qrels.jsonl")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
