"""onramp parent/child 토큰 **사이즈** sweep — 임시 인덱스 재색인 + page-level 검색 (#212).

`eval_chunking_ab.py`(#238/#247)는 splitter **종류** 비교용(onramp는 기본 사이즈 고정)이다.
이 스크립트는 onramp(SemanticChunker)의 **child/parent 토큰 사이즈를 바꿔가며** 검색 품질을
측정한다(사이즈 ablation). 한 변수만 바꾼다 — 예: child만 sweep, parent 고정.

설계 원칙(=eval_chunking_ab.py와 동일):
- production 인덱스 미접촉 — 임시 Qdrant 컬렉션 + 임시 OpenSearch 인덱스, 끝나면 삭제.
- mode: dense(Qdrant) · hybrid(Dense+BM25) · rerank(hybrid+리랭커=production 미러).
- domain 미적용 — 사이즈 효과만 격리.
- settings override로 색인·검색이 임시 인덱스를 가리키게 한다(코어 변경 없음).
- rerank+remote는 strict — preflight로 GPU 도달 확인, --require-reranker면 dense 폴백 시 fail-loud.

전제: 라이브 Qdrant + OpenSearch(hybrid/rerank) + Postgres(cleaned_markdown) + OpenAI 임베딩.
      rerank는 GPU 리랭커(RERANKER_BACKEND=remote·RERANKER_SERVICE_URL) 필요.
사용:
    python scripts/eval_size_sweep.py --child-target 256 --parent-target 1200 --mode rerank --require-reranker
    python scripts/eval_size_sweep.py --child-target 400 --parent-target 1200 --mode dense --doc-limit 50  # 소규모 먼저
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

import anyio  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db.models import SourceDocument  # noqa: E402
from app.db.postgres import session_scope  # noqa: E402
from app.eval.dataset import load_golden_set  # noqa: E402
from app.eval.metrics import aggregate, collapse_to_pages  # noqa: E402
from app.eval.retrieval_adapter import retrieve_for_eval  # noqa: E402
from app.eval.size_sweep import SizeConfig, chunk_page_sized, page_from_row  # noqa: E402
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


async def _reindex(config: SizeConfig, doc_limit: int | None) -> int:
    """코퍼스를 사이즈 config로 재청킹해 임시 인덱스(=override된 settings)에 색인."""
    pages = await _load_pages(doc_limit)
    children: list[ChildChunk] = []
    for page in pages:
        children.extend(chunk_page_sized(page, config.child_target, config.parent_target))
    logger.info(
        "재청킹: %d pages → %d child chunks (child=%d, parent=%d)",
        len(pages),
        len(children),
        config.child_target,
        config.parent_target,
    )
    if not children:
        raise RuntimeError("청크가 0개 — cleaned_markdown 코퍼스가 비었는지 확인하세요")
    await index_children(children)  # settings.qdrant_collection(=임시) + bm25면 settings.opensearch_index(=임시)
    return len(children)


async def _preflight_rerank(settings) -> None:
    """remote 리랭커 도달 확인 — 실패 시 fail-loud(전체 sweep이 조용히 dense 폴백되는 것 차단)."""
    from app.agents.retriever.rerank import get_reranker

    try:
        await anyio.to_thread.run_sync(get_reranker().rerank, "health check", [("hello world", {})])
    except Exception as exc:  # noqa: BLE001 — preflight는 어떤 실패든 중단
        raise SystemExit(
            f"리랭커 preflight 실패 (backend={settings.reranker_backend}): {exc}\n"
            "  → RERANKER_BACKEND=remote + URL(Redis 'reranker:service_url' 또는 env RERANKER_SERVICE_URL) 확인"
        ) from exc
    logger.info("리랭커 preflight OK (backend=%s)", settings.reranker_backend)


async def _eval_page_level(golden, *, mode: str, top_k: int, top_n: int, concurrency: int):
    """임시 인덱스를 mode로 **병렬** 검색해 page-level 매크로 평균 + rerank 발화율을 반환(domain 미적용)."""
    sem = asyncio.Semaphore(concurrency)

    async def _one(g):
        async with sem:
            r = await retrieve_for_eval(g.query, mode=mode, domains=None, top_k=top_k, top_n=top_n)
        return r, (collapse_to_pages(r.chunk_ids), set(g.page_ids))

    pairs = await asyncio.gather(*(_one(g) for g in golden))
    page_per_query = [p for _, p in pairs]
    # rerank 모드: tau_score(top_n 내 최대 raw)=0이면 dense 폴백 신호(운영 _vector_fallback raw=0.0).
    rerank_ratio = None
    if mode == "rerank" and pairs:
        rerank_ratio = sum(1 for r, _ in pairs if r.tau_score != 0.0) / len(pairs)
    return aggregate(page_per_query), rerank_ratio


async def _cleanup(collection: str, *, bm25: bool) -> None:
    """임시 Qdrant 컬렉션 + (bm25면) 임시 OpenSearch 인덱스를 삭제한다."""
    from app.db.qdrant import get_qdrant

    get_qdrant().delete_collection(collection_name=collection)
    if bm25:
        from app.db.opensearch import get_opensearch

        await get_opensearch().delete_index()
    logger.info("임시 인덱스 삭제: %s (Qdrant%s)", collection, " + OpenSearch" if bm25 else "")


async def run(args) -> int:
    config = SizeConfig(child_target=args.child_target, parent_target=args.parent_target)
    settings = get_settings()
    collection = config.collection_name(args.collection_prefix)
    top_k = args.top_k if args.top_k is not None else settings.retriever_top_k
    top_n = args.top_n if args.top_n is not None else settings.retriever_top_n
    bm25 = args.mode in ("hybrid", "rerank")  # hybrid/rerank는 BM25(OpenSearch) 후보가 필요

    # 임시 인덱스로 override (색인·검색이 모두 이 settings를 호출 시점에 읽는다 — 코어 변경 없음).
    settings.qdrant_collection = collection
    settings.bm25_search_enabled = bm25
    if bm25:  # 임시 OpenSearch 인덱스도 분리(ensure_index가 자동 생성, delete_index가 정리)
        settings.opensearch_index = collection
        settings.opensearch_index_v1 = f"{collection}-v1"
    logger.info(
        "임시 인덱스=%s · config_hash=%s · child=%d · parent=%d · mode=%s · reranker=%s",
        collection,
        config.hash,
        config.child_target,
        config.parent_target,
        args.mode,
        settings.reranker_backend if args.mode == "rerank" else "n/a",
    )

    if args.mode == "rerank" and settings.reranker_backend == "remote":
        await _preflight_rerank(settings)

    n_chunks = await _reindex(config, args.doc_limit)

    golden = load_golden_set(args.queries, args.qrels)
    summary, rerank_ratio = await _eval_page_level(
        golden, mode=args.mode, top_k=top_k, top_n=top_n, concurrency=args.concurrency
    )

    # rerank strict: 일부 질의라도 dense 폴백되면 A/B 오염 → 리포트 미발행(정리 후 중단).
    if args.require_reranker and args.mode == "rerank" and (rerank_ratio is None or rerank_ratio < 1.0):
        if not args.keep:
            await _cleanup(collection, bm25=bm25)
        raise SystemExit(
            f"리랭커 strict 위반: rerank_fired_ratio={rerank_ratio} < 1.0 — 일부 질의가 dense 폴백(오염). "
            "리포트 미발행. GPU/URL 상태 확인 후 재실행하세요."
        )

    print(f"\n=== onramp size sweep (page-level, {args.mode}, domain 미적용) ===")
    print(
        f"config_hash={config.hash}  collection={collection}  chunks={n_chunks}  "
        f"child_target={config.child_target}  parent_target={config.parent_target}"
    )
    if args.mode == "rerank":
        print(f"  reranker_backend={settings.reranker_backend}  rerank_fired_ratio={rerank_ratio}")
    for key, val in summary.as_dict().items():
        print(f"  {key:<14}: {val}")
    print(f"  n(page_ids 보유 평가질문) = {summary.n}")

    if not args.keep:
        await _cleanup(collection, bm25=bm25)
    else:
        logger.info("임시 인덱스 보존: %s (--keep)", collection)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="onramp parent/child 토큰 사이즈 sweep — page-level 검색 (#212).")
    parser.add_argument("--child-target", type=int, default=400, help="onramp child target 토큰")
    parser.add_argument("--parent-target", type=int, default=1200, help="onramp parent target 토큰")
    parser.add_argument(
        "--mode",
        default="dense",
        choices=["dense", "hybrid", "rerank"],
        help="검색 mode. dense=Qdrant, hybrid=Dense+BM25, rerank=hybrid+리랭커(production 미러, GPU 필요)",
    )
    parser.add_argument("--concurrency", type=int, default=8, help="검색 평가 병렬도(질의 동시 처리 수)")
    parser.add_argument(
        "--require-reranker", action="store_true", help="rerank 모드에서 dense 폴백이 한 건이라도 있으면 fail-loud"
    )
    parser.add_argument("--doc-limit", type=int, default=None, help="재색인 문서 수 제한(소규모 먼저)")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--collection-prefix", default="onramp-eval-size")
    parser.add_argument("--keep", action="store_true", help="끝나도 임시 인덱스를 삭제하지 않음")
    parser.add_argument("--queries", type=Path, default=ROOT_DIR / "data" / "eval" / "queries.jsonl")
    parser.add_argument("--qrels", type=Path, default=ROOT_DIR / "data" / "eval" / "qrels.jsonl")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
