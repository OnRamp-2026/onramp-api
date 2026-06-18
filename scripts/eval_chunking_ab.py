"""청킹 방식 A/B — config-hash 임시 컬렉션 재색인 + page-level 검색 게이트 (#212 step 5).

각 청킹 구성(onramp/token/markdown/recursive)을 `SourceDocument.cleaned_markdown`으로 재청킹해
**임시 Qdrant 컬렉션**(`onramp-eval-<strategy>-<hash>`)에 dense 색인하고, 골든셋으로 **splitter-독립
page-level** 검색 지표(#212 §2-2)를 측정한다.

설계 원칙:
- production 인덱스를 건드리지 않는다 — 임시 이름(Qdrant 컬렉션 + OpenSearch 인덱스) + 끝나면 삭제.
- **mode 선택**: dense(Qdrant) · hybrid(Dense+BM25) · **rerank(hybrid+리랭커=production 미러)**.
  청킹은 BM25와도 상호작용하므로(키워드 밀도·청크 길이), production 결정은 rerank로 본다.
- **domain 미적용**(domains=None) — 청킹 효과만 격리(도메인 가산은 별도 레버).
- settings 싱글톤 override로 색인·검색 모두 임시 인덱스를 가리키게 한다(코어 변경 없음).

전제: 라이브 Qdrant + OpenSearch(hybrid/rerank) + Postgres(cleaned_markdown) + OpenAI 임베딩.
      rerank mode는 GPU 리랭커(RERANKER_BACKEND=remote·RERANKER_SERVICE_URL)가 떠 있어야 한다.
사용:
    python scripts/eval_chunking_ab.py --strategy token --mode dense
    python scripts/eval_chunking_ab.py --strategy onramp --mode rerank   # production 미러(GPU 필요)
    python scripts/eval_chunking_ab.py --strategy recursive --doc-limit 50  # 소규모 먼저
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


async def _eval_page_level(golden, *, mode: str, top_k: int, top_n: int):
    """임시 인덱스를 주어진 mode로 검색해 page-level 매크로 평균을 반환(domain 미적용 — 청킹 격리).

    mode: dense(Qdrant) · hybrid(Dense+BM25 RRF) · rerank(hybrid 후보 + 리랭커, production 미러).
    """
    page_per_query: list[tuple[list[str], set[str]]] = []
    for g in golden:
        result = await retrieve_for_eval(g.query, mode=mode, domains=None, top_k=top_k, top_n=top_n)
        page_per_query.append((collapse_to_pages(result.chunk_ids), set(g.page_ids)))
    return aggregate(page_per_query)


async def _cleanup(collection: str, *, bm25: bool) -> None:
    """임시 Qdrant 컬렉션 + (bm25면) 임시 OpenSearch 인덱스를 삭제한다."""
    from app.db.qdrant import get_qdrant

    get_qdrant().delete_collection(collection_name=collection)
    if bm25:
        from app.db.opensearch import get_opensearch

        await get_opensearch().delete_index()
    logger.info("임시 인덱스 삭제: %s (Qdrant%s)", collection, " + OpenSearch" if bm25 else "")


async def run(args) -> int:
    config = ChunkingConfig(strategy=args.strategy, chunk_tokens=args.chunk_tokens, chunk_overlap=args.chunk_overlap)
    settings = get_settings()
    collection = config.collection_name(args.collection_prefix)
    top_k = args.top_k if args.top_k is not None else settings.retriever_top_k
    top_n = args.top_n if args.top_n is not None else settings.retriever_top_n
    bm25 = args.mode in ("hybrid", "rerank")  # hybrid/rerank는 BM25(OpenSearch) 후보가 필요

    # 임시 인덱스로 override (색인·검색이 모두 이 settings를 호출 시점에 읽는다 — 코어 변경 없음).
    settings.qdrant_collection = collection
    settings.bm25_search_enabled = bm25
    if bm25:  # 임시 OpenSearch 인덱스도 분리(ensure_index가 자동 생성)
        settings.opensearch_index = collection
        settings.opensearch_index_v1 = f"{collection}-v1"
    logger.info(
        "임시 인덱스=%s · config_hash=%s · mode=%s · reranker=%s",
        collection,
        config.hash,
        args.mode,
        settings.reranker_backend if args.mode == "rerank" else "n/a",
    )

    n_chunks = await _reindex(config, args.doc_limit)

    golden = load_golden_set(args.queries, args.qrels)
    summary = await _eval_page_level(golden, mode=args.mode, top_k=top_k, top_n=top_n)

    print(f"\n=== 청킹 A/B (page-level, {args.mode}, domain 미적용) — {args.strategy} ===")
    print(f"config_hash={config.hash}  collection={collection}  chunks={n_chunks}")
    for key, val in summary.as_dict().items():
        print(f"  {key:<14}: {val}")
    print(f"  n(page_ids 보유 평가질문) = {summary.n}")

    if not args.keep:
        await _cleanup(collection, bm25=bm25)
    else:
        logger.info("임시 인덱스 보존: %s (--keep)", collection)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="청킹 방식 A/B — 임시 컬렉션 재색인 + page-level 게이트 (#212).")
    parser.add_argument("--strategy", required=True, choices=["onramp", "token", "markdown", "recursive"])
    parser.add_argument(
        "--mode",
        default="dense",
        choices=["dense", "hybrid", "rerank"],
        help="검색 mode. dense=Qdrant, hybrid=Dense+BM25, rerank=hybrid+리랭커(production 미러, GPU 필요)",
    )
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
