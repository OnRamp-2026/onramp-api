"""도메인 필터 ON/OFF 검색 격차 측정 (이슈 #65).

단일 도메인 하드 필터가 멀티 도메인(장애 대응·온보딩 등) 질문의 recall을 얼마나
깎는지 정량화한다. 두 가지 분석을 출력한다.

  [A] 구조적 분석 (결정론, 오프라인 — Qdrant 불필요)
      골든셋의 `gold_domains`(정답이 걸친 도메인 집합)만으로 멀티 도메인 질문을 판정하고,
      하드 필터(domain=g.domain)가 도달 불가능하게 만드는 정답 **도메인** 비율(=recall 상한
      손실)을 집계한다. 단일 도메인 하드 필터가 멀티 도메인 정답을 구조적으로 배제하는 양.
  [A'] 드리프트 검증 (Qdrant 필요, 선택)
      저장된 `gold_domains` 가 실제 색인 청크 도메인(payload.domain 역참조)과 어긋나면 경고.
      골든 라벨이 색인 재라벨링으로 stale 됐는지 잡는다.
  [B] 실측 비교 (라이브 검색)
      같은 질문을 filter ON(domain=g.domain) vs OFF(domain=None)로 검색해 recall@k 등을
      비교한다. 정답이 멀티 도메인인 질문을 따로 묶어 격차를 본다.

[A']·[B]는 라이브 Qdrant(+[B]는 OpenAI 임베딩·리랭커)를 사용한다(비용·인프라 필요).

사용:
    python scripts/eval_domain_filter.py                  # 구조적 + 드리프트 + rerank 실측
    python scripts/eval_domain_filter.py --mode dense     # 실측을 dense로 (빠름)
    python scripts/eval_domain_filter.py --structural-only # [A]만 (Qdrant·임베딩·리랭커 불필요)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings  # noqa: E402
from app.db.qdrant import get_qdrant  # noqa: E402
from app.eval.dataset import GoldenQuery, load_golden_set  # noqa: E402
from app.eval.metrics import recall_at_k  # noqa: E402
from app.eval.retrieval_adapter import Mode, retrieve_for_eval  # noqa: E402

logger = logging.getLogger(__name__)

UNKNOWN_DOMAIN = "<unknown>"  # 골든 청크가 인덱스에 없을 때(stale 라벨)


def _point_id(chunk_id: str) -> str:
    """indexer._point_id 와 동일 — chunk_id → Qdrant point UUID5 (멱등)."""
    return str(uuid5(NAMESPACE_URL, chunk_id))


def lookup_chunk_domains(chunk_ids: set[str], *, collection: str) -> dict[str, str]:
    """정답 chunk_id → payload.domain 매핑을 Qdrant에서 역참조한다.

    인덱스에 없는 chunk_id는 UNKNOWN_DOMAIN 으로 둔다(stale 골든 신호).
    """
    if not chunk_ids:
        return {}
    ids = sorted(chunk_ids)
    client = get_qdrant()
    points = client.retrieve(
        collection_name=collection,
        ids=[_point_id(c) for c in ids],
        with_payload=True,
        with_vectors=False,
    )
    # payload.chunk_id 로 되돌려 매핑(retrieve 순서·결측에 견고)
    by_chunk: dict[str, str] = {}
    for p in points:
        payload = p.payload or {}
        cid = payload.get("chunk_id")
        if cid:
            by_chunk[cid] = payload.get("domain") or UNKNOWN_DOMAIN
    return {c: by_chunk.get(c, UNKNOWN_DOMAIN) for c in ids}


# ---------------------------------------------------------------------------
# [A] 구조적 분석 — gold_domains 기반, 오프라인(Qdrant 불필요)
# ---------------------------------------------------------------------------


def _excluded_domain_ratio(gold_domains: tuple[str, ...], filter_domain: str | None) -> float:
    """하드 필터(filter_domain)가 도달 못 하는 정답 **도메인** 비율(=recall 상한 손실).

    filter_domain 이 None(무필터)이면 0.0. 그 외엔 gold_domains 중 filter_domain 아닌 비율.
    예: gold_domains=[incident, api_reference], filter=incident → 0.5 (api_reference 배제).
    """
    if not gold_domains or filter_domain is None:
        return 0.0
    excluded = sum(1 for d in gold_domains if d != filter_domain)
    return excluded / len(gold_domains)


def structural_analysis(golden: list[GoldenQuery]) -> dict:
    """gold_domains로 멀티 도메인 판정 + 하드 필터의 구조적 recall 상한 손실을 집계(오프라인)."""
    rows = []
    for g in golden:
        if not g.relevant_chunk_ids:  # unanswerable 제외
            continue
        rows.append(
            {
                "qid": g.qid,
                "query_domain": g.domain,
                "n_relevant": len(g.relevant_chunk_ids),
                "answer_domains": list(g.gold_domains),
                "n_answer_domains": len(g.gold_domains),
                "is_multi_domain": g.is_multi_domain,
                "excluded_ratio": _excluded_domain_ratio(g.gold_domains, g.domain),
            }
        )
    return {"rows": rows}


def drift_check(golden: list[GoldenQuery], *, collection: str) -> list[dict]:
    """저장된 gold_domains vs 실제 색인 청크 도메인(Qdrant 역참조) 불일치를 찾는다.

    골든 라벨이 색인 재라벨링으로 stale 됐는지 잡는 안전망. Qdrant 필요.
    """
    drifts = []
    for g in golden:
        rel = set(g.relevant_chunk_ids)
        if not rel:
            continue
        rel_domains = lookup_chunk_domains(rel, collection=collection)
        actual = set(rel_domains.values())
        has_unknown = UNKNOWN_DOMAIN in actual
        actual_known = actual - {UNKNOWN_DOMAIN}
        gold = set(g.gold_domains)
        if has_unknown or actual_known != gold:
            drifts.append(
                {
                    "qid": g.qid,
                    "gold_domains": sorted(gold),
                    "actual_domains": sorted(actual_known),
                    "has_unknown": has_unknown,
                }
            )
    return drifts


# ---------------------------------------------------------------------------
# [B] 실측 — filter ON vs OFF
# ---------------------------------------------------------------------------


async def empirical_recall(golden: list[GoldenQuery], *, mode: Mode, top_k, top_n, structural_rows) -> list[dict]:
    """질문별 recall@top_n 을 filter ON(g.domain) / OFF(None) 두 조건으로 측정."""
    multi_map = {r["qid"]: r["is_multi_domain"] for r in structural_rows}
    out = []
    for g in golden:
        rel = set(g.relevant_chunk_ids)
        if not rel:
            continue
        on = await retrieve_for_eval(g.query, mode=mode, domain=g.domain, top_k=top_k, top_n=top_n)
        off = await retrieve_for_eval(g.query, mode=mode, domain=None, top_k=top_k, top_n=top_n)
        out.append(
            {
                "qid": g.qid,
                "is_multi_domain": multi_map.get(g.qid, False),
                "recall_on": recall_at_k(on.chunk_ids, rel, top_n),
                "recall_off": recall_at_k(off.chunk_ids, rel, top_n),
            }
        )
    return out


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def print_structural(rows: list[dict]) -> None:
    n = len(rows)
    multi = [r for r in rows if r["is_multi_domain"]]
    excluded_pos = [r for r in rows if r["excluded_ratio"] > 0]

    print("\n" + "=" * 64)
    print("[A] 구조적 분석 — 하드 필터의 recall 상한 손실 (gold_domains, 오프라인)")
    print("=" * 64)
    print(f"평가 대상(answerable) 질문         : {n}")
    print(f"정답이 멀티 도메인인 질문           : {len(multi)}  ({_mean([r['is_multi_domain'] for r in rows]):.1%})")
    print(f"필터가 정답 도메인 일부를 배제하는 질문: {len(excluded_pos)}  (excluded_ratio>0)")
    print(f"평균 배제 도메인 비율(전체)         : {_mean([r['excluded_ratio'] for r in rows]):.4f}")
    print(f"평균 배제 도메인 비율(멀티 도메인군) : {_mean([r['excluded_ratio'] for r in multi]):.4f}")

    if multi:
        print("\n  멀티 도메인 정답 질문 (필터가 깎는 후보):")
        print(f"  {'qid':<8}{'query_domain':<14}{'answer_domains':<34}{'excl_ratio':>10}")
        for r in sorted(multi, key=lambda x: -x["excluded_ratio"]):
            print(
                f"  {r['qid']:<8}{str(r['query_domain']):<14}"
                f"{','.join(r['answer_domains']):<34}{r['excluded_ratio']:>10.3f}"
            )


def print_drift(drifts: list[dict]) -> None:
    if not drifts:
        print("\n[A'] 드리프트 검증 — gold_domains == 실제 색인 도메인 (OK, 불일치 없음)")
        return
    print("\n" + "=" * 64)
    print(f"[A'] ⚠ 드리프트 {len(drifts)}건 — gold_domains 와 실제 색인 청크 도메인 불일치")
    print("=" * 64)
    print(f"  {'qid':<8}{'gold_domains':<28}{'actual_domains':<28}{'unknown':>8}")
    for d in drifts:
        print(
            f"  {d['qid']:<8}{','.join(d['gold_domains']):<28}"
            f"{','.join(d['actual_domains']):<28}{str(d['has_unknown']):>8}"
        )
    print("  → 골든 gold_domains 라벨 또는 색인 도메인을 재검토하세요(stale 가능).")


def print_empirical(rows: list[dict], *, mode: str, top_n: int) -> None:
    overall_on = _mean([r["recall_on"] for r in rows])
    overall_off = _mean([r["recall_off"] for r in rows])
    multi = [r for r in rows if r["is_multi_domain"]]
    single = [r for r in rows if not r["is_multi_domain"]]

    print("\n" + "=" * 64)
    print(f"[B] 실측 — filter ON(g.domain) vs OFF(None)  [mode={mode}, recall@{top_n}]")
    print("=" * 64)
    print(f"{'group':<22}{'n':>5}{'ON':>10}{'OFF':>10}{'Δ(OFF-ON)':>12}")
    print("-" * 59)
    for label, grp in [("전체", rows), ("멀티 도메인 정답", multi), ("단일 도메인 정답", single)]:
        on = _mean([r["recall_on"] for r in grp])
        off = _mean([r["recall_off"] for r in grp])
        print(f"{label:<22}{len(grp):>5}{on:>10.4f}{off:>10.4f}{off - on:>+12.4f}")
    print("\n해석: Δ가 양(+)이면 도메인 필터가 recall을 깎고 있다는 신호. "
          "멀티 도메인군에서 Δ가 클수록 이슈 #65의 가설이 강해진다.")
    _ = (overall_on, overall_off)


async def run(args) -> int:
    golden = load_golden_set(args.queries, args.qrels)
    settings = get_settings()
    collection = settings.qdrant_collection
    logger.info("골든셋 %d개 로드", len(golden))

    structural = structural_analysis(golden)
    print_structural(structural["rows"])

    if args.structural_only:
        return 0

    # [A'] 드리프트 검증 — gold_domains vs 실제 색인 도메인 (Qdrant 필요)
    print_drift(drift_check(golden, collection=collection))

    top_k = args.top_k if args.top_k is not None else settings.retriever_top_k
    top_n = args.top_n if args.top_n is not None else settings.retriever_top_n
    empirical = await empirical_recall(
        golden, mode=args.mode, top_k=top_k, top_n=top_n, structural_rows=structural["rows"]
    )
    print_empirical(empirical, mode=args.mode, top_n=top_n)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="도메인 필터 ON/OFF 검색 격차 측정 (이슈 #65).")
    parser.add_argument("--queries", type=Path, default=ROOT_DIR / "data" / "eval" / "queries.jsonl")
    parser.add_argument("--qrels", type=Path, default=ROOT_DIR / "data" / "eval" / "qrels.jsonl")
    parser.add_argument("--mode", choices=["dense", "rerank"], default="rerank", help="실측 검색 모드")
    parser.add_argument("--top-k", type=int, default=None, help="Qdrant 후보 풀 (기본: config)")
    parser.add_argument("--top-n", type=int, default=None, help="최종 top-N (기본: config)")
    parser.add_argument(
        "--structural-only", action="store_true", help="[A] 구조적 분석만 (임베딩/리랭커 불필요)"
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
