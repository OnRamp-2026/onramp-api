"""도메인 필터 모드(hard/hybrid/soft) A/B — 골든셋으로 검색 지표를 모드별 비교.

전제: `docker compose up`(Qdrant) + 색인 데이터 + OPENAI_API_KEY. CI 비포함(수동 실행).
사용:
    python scripts/eval_filter_modes.py [--modes hard,hybrid,soft] [--rank rerank]
"""

from __future__ import annotations

import argparse
import asyncio

from app.agents.retriever.search import FilterMode
from app.eval.dataset import load_golden_set
from app.eval.metrics import MetricSummary, aggregate
from app.eval.retrieval_adapter import Mode, ranked_chunk_ids

FILTER_MODES: tuple[FilterMode, ...] = ("hard", "hybrid", "soft")


async def _eval_filter_mode(golden, filter_mode: FilterMode, rank_mode: Mode) -> MetricSummary:
    per_query: list[tuple[list[str], set[str]]] = []
    for g in golden:
        ranked = await ranked_chunk_ids(g.query, mode=rank_mode, domains=[g.domain] if g.domain else None, filter_mode=filter_mode)
        per_query.append((ranked, set(g.relevant_chunk_ids)))
    return aggregate(per_query)


async def main_async(modes: list[FilterMode], rank_mode: Mode) -> None:
    golden = load_golden_set()
    print(f"골든셋 {len(golden)}건 · 랭킹={rank_mode}\n")
    summaries: dict[FilterMode, MetricSummary] = {}
    for filter_mode in modes:
        summary = await _eval_filter_mode(golden, filter_mode, rank_mode)
        summaries[filter_mode] = summary
        metrics = "  ".join(f"{k}={v}" for k, v in summary.as_dict().items())
        print(f"[{filter_mode:<6}] n={summary.n}  {metrics}")

    best = max(summaries, key=lambda m: summaries[m].hit_rate)
    print(f"\n→ Hit 기준 최고 모드: {best}")


def main() -> None:
    parser = argparse.ArgumentParser(description="도메인 필터 모드(hard/hybrid/soft) A/B")
    parser.add_argument("--modes", default="hard,hybrid,soft", help="비교할 필터 모드 (쉼표 구분)")
    parser.add_argument("--rank", default="rerank", choices=["dense", "rerank"], help="랭킹 방식")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes:
        parser.error("--modes에 최소 1개 필터 모드가 필요합니다 (가능: " + ", ".join(FILTER_MODES) + ")")
    invalid = [m for m in modes if m not in FILTER_MODES]
    if invalid:
        parser.error(f"지원하지 않는 필터 모드: {invalid} (가능: {list(FILTER_MODES)})")
    asyncio.run(main_async(modes, args.rank))  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
