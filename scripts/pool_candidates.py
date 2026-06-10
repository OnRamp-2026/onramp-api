"""qrels 라벨 누락 방지용 pooling 보조 — 질문별 rerank top-N 후보를 사이드카로 출력한다.

큰 코퍼스에서는 같은 질문에 답이 되는 청크가 여러 페이지에 흩어질 수 있어(중복 문서·튜토리얼/
레퍼런스 중복), 출처 청크 하나만 라벨하면 검색기가 *정답인데 라벨 안 된* 청크를 1위로 올려도
오답 처리된다(체계적 과소평가). IR 표준 pooling: 검색 top-N을 사람이 보고 qrels를 보완한다.

사용:
    python scripts/pool_candidates.py --queries data/eval/queries.multi-hop.draft.jsonl
    → data/eval/pool.multi-hop.draft.jsonl (qid별 top-N chunk_id·제목·점수)

의존: 라이브 Qdrant + OpenAI 임베딩 + 리랭커 (비용 소액).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import anyio  # noqa: E402

from app.agents.retriever.rerank import apply_metadata_weight, get_reranker  # noqa: E402
from app.agents.retriever.search import dense_search  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.rag.embedder import get_embedder  # noqa: E402

logger = logging.getLogger(__name__)


async def _top_candidates(query: str, domain: str | None, *, top_k: int, top_n: int) -> list[dict]:
    """rerank 상위 후보 (chunk_id·제목·점수) — retrieval_adapter 경로 미러 + 메타 노출."""
    settings = get_settings()
    qvec = await get_embedder().embed_query(query)
    hits = await dense_search(qvec, top_k, domain=domain, settings=settings)
    if not hits and domain:
        hits = await dense_search(qvec, top_k, domain=None, settings=settings)
    candidates = [(p.payload.get("content", ""), p.payload or {}) for p in hits]
    try:
        reranked = await anyio.to_thread.run_sync(get_reranker().rerank, query, candidates)
        ranked = [(apply_metadata_weight(score, payload, settings), payload) for score, payload in reranked]
        ranked.sort(key=lambda item: item[0], reverse=True)
    except Exception:
        logger.warning("리랭커 실패 — vector score 순 폴백", exc_info=True)
        ranked = [(p.score, p.payload or {}) for p in hits]
        ranked.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "chunk_id": payload.get("chunk_id", ""),
            "page_title": payload.get("page_title", ""),
            "domain": payload.get("domain", ""),
            "score": round(float(score), 4),
        }
        for score, payload in ranked[:top_n]
        if payload.get("chunk_id")
    ]


async def run(args) -> None:
    rows = [json.loads(line) for line in args.queries.read_text(encoding="utf-8").splitlines() if line.strip()]
    out_rows = []
    for i, row in enumerate(rows, start=1):
        qid, query = row.get("qid"), row.get("query")
        if not qid or not query:  # 깨진 레코드 하나가 전체 배치를 중단시키지 않도록
            logger.warning("[%d/%d] 스킵: qid/query 누락", i, len(rows))
            continue
        if not row.get("is_answerable", True):  # unanswerable은 pooling 불필요
            continue
        logger.info("[%d/%d] pooling: %.50s", i, len(rows), query)
        cands = await _top_candidates(query, row.get("domain"), top_k=args.top_k, top_n=args.top_n)
        out_rows.append({"qid": qid, "query": query, "candidates": cands})

    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n", encoding="utf-8")
    logger.info(
        "pooling %d건 → %s (검수: 후보 중 정답인데 qrels에 없는 chunk_id를 보완하세요)", len(out_rows), args.out
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="질문별 rerank top-N 후보 출력 (qrels 보완 검수용).")
    parser.add_argument("--queries", type=Path, required=True, help="queries jsonl (draft 또는 확정)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.out is None:
        name = args.queries.name.replace("queries", "pool", 1)
        if not name.endswith(".draft.jsonl"):
            name = name.removesuffix(".jsonl") + ".draft.jsonl"  # gitignore 패턴 유지
        args.out = args.queries.parent / name

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
