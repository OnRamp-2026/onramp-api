"""청킹 방식 A/B — config-hash 임시 컬렉션 재색인 + page-level 검색 게이트 (#212 step 5).

각 청킹 구성(onramp/token/markdown/recursive)을 `SourceDocument.cleaned_markdown`으로 재청킹해
**임시 Qdrant(+선택적 OpenSearch) 컬렉션**에 색인하고, 골든셋으로 **splitter-독립 page-level**
검색 지표(#212 §2-2)를 측정한다.

검색 모드(--mode):
- `dense`(기본)  — Qdrant kNN 단독. OpenSearch 불필요(=기존 동작).
- `sparse`       — OpenSearch BM25 단독.
- `hybrid`       — Dense+BM25 RRF (리랭커 없음).
- `rerank`       — 운영 경로 미러: 1차 검색(HYBRID_SEARCH_ENABLED면 hybrid) + remote Cross-Encoder 재정렬.

설계 원칙:
- production 인덱스를 건드리지 않는다 — Qdrant 임시 컬렉션 + **OpenSearch 임시 인덱스**(인덱스명 override)
  를 쓰고, 끝나면 삭제(--keep로 보존).
- settings 싱글톤 override로 색인·검색이 모두 임시 인덱스를 호출 시점에 읽게 한다(코어 변경 없음).
- **domain 미적용**(domains=None) — 청킹 효과만 격리(도메인 가산은 별도 레버).
- BM25 모드는 `bm25_search_enabled=True`로 색인·검색 모두 OpenSearch를 태운다.
- `rerank` 모드 + `reranker_backend=remote`는 **strict** — preflight로 GPU 도달을 확인하고,
  --require-reranker면 일부 질의라도 dense 폴백 시 fail-loud(리포트 미발행, A/B 오염 차단, #212 §2-5).

전제: 라이브 Qdrant + Postgres(cleaned_markdown) + OpenAI 임베딩 (+BM25 모드면 OpenSearch, rerank면 remote 리랭커).
사용:
    python scripts/eval_chunking_ab.py --strategy onramp --child-target 256 --parent-target 1200            # dense
    python scripts/eval_chunking_ab.py --strategy onramp --child-target 256 --mode rerank --require-reranker  # hybrid+rerank
    python scripts/eval_chunking_ab.py --strategy onramp --doc-limit 50 --mode rerank                          # 소규모 먼저
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
from app.eval.chunking_experiment import ONRAMP, ChunkingConfig, chunk_page, page_from_row  # noqa: E402
from app.eval.dataset import load_golden_set  # noqa: E402
from app.eval.metrics import aggregate, collapse_to_pages  # noqa: E402
from app.eval.retrieval_adapter import Mode, retrieve_for_eval  # noqa: E402
from app.rag.chunker import ChildChunk  # noqa: E402
from app.rag.indexer import index_children  # noqa: E402

logger = logging.getLogger(__name__)

# 1차 검색에 BM25(OpenSearch)가 필요한 모드 — 임시 OpenSearch 인덱스 격리가 필요.
_BM25_MODES: frozenset[str] = frozenset({"sparse", "hybrid", "rerank"})


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
    """코퍼스를 config 전략으로 재청킹해 임시 인덱스(=override된 settings)에 색인.

    BM25 모드면 index_children이 Qdrant + OpenSearch 둘 다 색인한다(settings.bm25_search_enabled 기준).
    """
    pages = await _load_pages(doc_limit)
    children: list[ChildChunk] = []
    for page in pages:
        children.extend(chunk_page(config, page))
    logger.info("재청킹: %d pages → %d child chunks (전략=%s)", len(pages), len(children), config.strategy)
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


async def _eval_page_level(golden, *, mode: Mode, top_k: int, top_n: int, concurrency: int):
    """임시 인덱스를 mode로 병렬 검색해 page-level 매크로 평균 + rerank 발화율을 반환(domain 미적용)."""
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
        fired = sum(1 for r, _ in pairs if r.tau_score != 0.0)
        rerank_ratio = fired / len(pairs)
    return aggregate(page_per_query), rerank_ratio


async def _cleanup(collection: str, needs_bm25: bool, concrete_os_index: str) -> None:
    from app.db.qdrant import get_qdrant

    get_qdrant().delete_collection(collection_name=collection)
    logger.info("임시 Qdrant 컬렉션 삭제: %s", collection)
    if needs_bm25:
        from app.db.opensearch import get_opensearch

        await get_opensearch().delete_index(concrete_os_index)
        logger.info("임시 OpenSearch 인덱스 삭제: %s", concrete_os_index)


async def run(args) -> int:
    mode: Mode = args.mode
    config = ChunkingConfig(
        strategy=args.strategy,
        chunk_tokens=args.chunk_tokens,
        chunk_overlap=args.chunk_overlap,
        child_target=args.child_target,
        parent_target=args.parent_target,
    )
    settings = get_settings()
    collection = config.collection_name(args.collection_prefix)
    top_k = args.top_k if args.top_k is not None else settings.retriever_top_k
    top_n = args.top_n if args.top_n is not None else settings.retriever_top_n
    needs_bm25 = mode in _BM25_MODES
    concrete_os_index = f"{collection}-v1"

    # 임시 인덱스로 override (색인·검색이 모두 이 settings를 호출 시점에 읽는다).
    settings.qdrant_collection = collection
    if needs_bm25:
        settings.bm25_search_enabled = True
        settings.opensearch_index = collection  # alias
        settings.opensearch_index_v1 = concrete_os_index  # concrete(자동 생성)
        logger.info("임시 인덱스 Qdrant=%s · OpenSearch(alias=%s) · mode=%s", collection, collection, mode)
    else:
        settings.bm25_search_enabled = False  # dense-only
        logger.info("임시 인덱스 Qdrant=%s · dense-only · mode=%s", collection, mode)

    if mode == "rerank" and settings.reranker_backend == "remote":
        await _preflight_rerank(settings)

    n_chunks = await _reindex(config, args.doc_limit)

    golden = load_golden_set(args.queries, args.qrels)
    summary, rerank_ratio = await _eval_page_level(
        golden, mode=mode, top_k=top_k, top_n=top_n, concurrency=args.concurrency
    )

    # rerank strict: 일부 질의라도 dense 폴백되면 A/B 오염 → 리포트 미발행(정리 후 중단).
    if args.require_reranker and mode == "rerank" and (rerank_ratio is None or rerank_ratio < 1.0):
        if not args.keep:
            await _cleanup(collection, needs_bm25, concrete_os_index)
        raise SystemExit(
            f"리랭커 strict 위반: rerank_fired_ratio={rerank_ratio} < 1.0 — 일부 질의가 dense 폴백(오염). "
            "리포트 미발행. GPU/URL 상태 확인 후 재실행하세요."
        )

    if args.strategy != ONRAMP and (args.child_target is not None or args.parent_target is not None):
        logger.warning("--child-target/--parent-target 은 onramp 전략에만 적용됩니다 (현재 %s — 무시)", args.strategy)

    print(f"\n=== 청킹 A/B (page-level, mode={mode}, domain 미적용) — {args.strategy} ===")
    size = (
        f"  child_target={args.child_target or 400}  parent_target={args.parent_target or 1200}"
        if args.strategy == ONRAMP
        else ""
    )
    print(f"config_hash={config.hash}  collection={collection}  chunks={n_chunks}{size}")
    if mode == "rerank":
        print(f"  reranker_backend={settings.reranker_backend}  rerank_fired_ratio={rerank_ratio}")
    for key, val in summary.as_dict().items():
        print(f"  {key:<14}: {val}")
    print(f"  n(page_ids 보유 평가질문) = {summary.n}")

    if not args.keep:
        await _cleanup(collection, needs_bm25, concrete_os_index)
        logger.info("임시 인덱스 정리 완료 (보존하려면 --keep)")
    else:
        logger.info(
            "임시 인덱스 보존: Qdrant=%s%s", collection, f" · OpenSearch={concrete_os_index}" if needs_bm25 else ""
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="청킹 방식 A/B — 임시 인덱스 재색인 + page-level 게이트 (#212).")
    parser.add_argument("--strategy", required=True, choices=["onramp", "token", "markdown", "recursive"])
    parser.add_argument(
        "--mode", choices=["dense", "sparse", "hybrid", "rerank"], default="dense", help="검색 모드 (기본 dense)"
    )
    parser.add_argument("--chunk-tokens", type=int, default=400, help="비교군 splitter 청크 토큰 크기(onramp 무관)")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="비교군 splitter 오버랩(onramp 무관)")
    parser.add_argument(
        "--child-target", type=int, default=None, help="onramp child target 토큰(사이즈 sweep). 미지정=기본 400"
    )
    parser.add_argument(
        "--parent-target", type=int, default=None, help="onramp parent target 토큰(사이즈 sweep). 미지정=기본 1200"
    )
    parser.add_argument("--concurrency", type=int, default=8, help="검색 평가 병렬도(질의 동시 처리 수)")
    parser.add_argument(
        "--require-reranker", action="store_true", help="rerank 모드에서 dense 폴백이 한 건이라도 있으면 fail-loud"
    )
    parser.add_argument("--doc-limit", type=int, default=None, help="재색인 문서 수 제한(소규모 먼저)")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--collection-prefix", default="onramp-eval")
    parser.add_argument("--keep", action="store_true", help="끝나도 임시 인덱스를 삭제하지 않음")
    parser.add_argument("--queries", type=Path, default=ROOT_DIR / "data" / "eval" / "queries.jsonl")
    parser.add_argument("--qrels", type=Path, default=ROOT_DIR / "data" / "eval" / "qrels.jsonl")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
