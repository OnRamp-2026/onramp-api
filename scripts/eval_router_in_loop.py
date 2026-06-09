"""router-in-the-loop A/B — 라우터 예측 도메인으로 검색해 필터 모드를 비교한다.

오라클(골든 정답 도메인) vs 라우터(예측 도메인)를 같은 실행에서 나란히 측정해,
"라우터가 도메인을 틀릴 때 hard/hybrid/soft가 얼마나 무너지나"를 본다. 랭킹은 dense
(리랭커 CPU 병목 회피). + 라우터 도메인 정확도·범위밖 차단도 함께 낸다.

전제: docker compose up(Qdrant) + 색인 데이터 + OPENAI_API_KEY. CI 비포함(수동 실행).
사용: python scripts/eval_router_in_loop.py [--modes hard,hybrid,soft]
"""

from __future__ import annotations

import argparse
import asyncio

from app.agents.retriever.search import FilterMode
from app.agents.router.node import route_node
from app.agents.state import Domain, UseCase
from app.eval.dataset import load_golden_set
from app.eval.metrics import MetricSummary, aggregate
from app.eval.retrieval_adapter import ranked_chunk_ids

FILTER_MODES: tuple[FilterMode, ...] = ("hard", "hybrid", "soft")


def _domain_str(domain: Domain | str | None) -> str | None:
    if domain is None:
        return None
    return domain.value if isinstance(domain, Domain) else domain


async def main_async(modes: list[FilterMode]) -> None:
    golden = load_golden_set()
    # per_query[source][mode] = list[(ranked_ids, relevant_set)]
    oracle: dict[FilterMode, list[tuple[list[str], set[str]]]] = {m: [] for m in modes}
    router: dict[FilterMode, list[tuple[list[str], set[str]]]] = {m: [] for m in modes}

    domain_total = domain_correct = 0
    unanswerable_total = unanswerable_blocked = 0

    for g in golden:
        relevant = set(g.relevant_chunk_ids)

        # 라우터 1회 실행 → 예측 도메인 / use_case
        routed = await route_node({"query": g.query})
        pred_domain = _domain_str(routed.get("domain"))
        blocked = routed.get("use_case") == UseCase.UNANSWERABLE

        # 라우터 도메인 정확도 (answerable + 골든 도메인 보유)
        if g.is_answerable and g.domain is not None:
            domain_total += 1
            domain_correct += pred_domain == g.domain
        # 범위밖 차단 정확도
        if not g.is_answerable:
            unanswerable_total += 1
            unanswerable_blocked += blocked

        for mode in modes:
            # 오라클: 골든 도메인으로 검색
            oracle[mode].append(
                (await ranked_chunk_ids(g.query, mode="dense", domain=g.domain, filter_mode=mode), relevant)
            )
            # 라우터: 예측 도메인으로 검색 (UNANSWERABLE이면 검색 생략 → 빈 결과)
            ids = [] if blocked else await ranked_chunk_ids(g.query, mode="dense", domain=pred_domain, filter_mode=mode)
            router[mode].append((ids, relevant))

    _print_block("오라클 도메인 (골든 정답)", oracle, modes)
    _print_block("router-in-the-loop (라우터 예측)", router, modes)
    if domain_total:
        print(f"\n라우터 도메인 정확도: {domain_correct}/{domain_total} ({domain_correct / domain_total:.3f})")
    if unanswerable_total:
        print(f"범위밖 차단: {unanswerable_blocked}/{unanswerable_total}")


def _print_block(title: str, data: dict[FilterMode, list[tuple[list[str], set[str]]]], modes: list[FilterMode]) -> None:
    print(f"\n=== {title} · 랭킹=dense ===")
    summaries: dict[FilterMode, MetricSummary] = {m: aggregate(data[m]) for m in modes}
    for mode in modes:
        metrics = "  ".join(f"{k}={v}" for k, v in summaries[mode].as_dict().items())
        print(f"[{mode:<6}] n={summaries[mode].n}  {metrics}")
    best = max(modes, key=lambda m: summaries[m].hit_rate)
    print(f"→ Hit 기준 최고: {best}")


def main() -> None:
    parser = argparse.ArgumentParser(description="router-in-the-loop 필터 모드 A/B")
    parser.add_argument("--modes", default="hard,hybrid,soft", help="비교할 필터 모드 (쉼표 구분)")
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    invalid = [m for m in modes if m not in FILTER_MODES]
    if invalid:
        parser.error(f"지원하지 않는 필터 모드: {invalid} (가능: {list(FILTER_MODES)})")
    asyncio.run(main_async(modes))  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
