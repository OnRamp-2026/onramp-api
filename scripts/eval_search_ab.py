"""검색 A/B — 질의 단일(A) vs 멀티(B) 도메인 가산 효과 격리 측정.

설계(공정성 핵심):
  - 같은 예측 캐시 재사용(라우터 재호출 X) → A=predicted_domains[:1], B=predicted_domains[:2].
  - 질문당 **검색+리랭크 1회**로 도메인 가산 전 base를 확보하고, 같은 base에 A/B 도메인 가산만
    각각 적용·정렬·평가 → **secondary 사용 여부만** 달라진다(검색/리랭크 2배·환경변동 제거).
  - 공식 그룹 = len(router_domains) 1/2. gold_domains는 보조 분석.

범위(중요): 검색에 **골든 질의 원문**을 쓴다(운영 라우터의 refined_query 아님). 따라서 secondary 도메인
  가산 효과를 격리한 유효 실험이지만 "운영 라우터 전체 효과"는 아니다. 운영 경로 A/B는 후속(캐시에 refined_query 저장).

전제: Qdrant + 임베딩(질문당 1회) 필요(라이브 검색). 예측 캐시는 전체가 신선해야 한다(아니면 비정상 종료).
사용:
    python scripts/eval_search_ab.py                       # 측정 → 콘솔
    python scripts/eval_search_ab.py --write-result        # + 결과 JSON 저장
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.eval.dataset import GoldenQuery, load_golden_set
from app.eval.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from app.eval.retrieval_adapter import base_soft_candidates, rank_chunk_ids_from_base
from app.eval.router_cache import current_meta, git_commit_sha, is_fresh, load_cache, sha12

_ROOT = Path(__file__).resolve().parents[1]
_QUERIES = _ROOT / "data/eval/queries.jsonl"
_QRELS = _ROOT / "data/eval/qrels.jsonl"
_DEFAULT_CACHE = str(_ROOT / ".cache/onramp-eval/router_predictions.jsonl")
_DEFAULT_RESULT = str(_ROOT / "data/eval/results/search_ab.json")

TOP_K = 20  # 후보 수 (도메인 무관 — A/B 공유)
TOP_N = 10  # 평가 반환 (@5·@10 둘 다 산출하려면 ≥10)
_METRIC_KEYS = ("hit@5", "recall@5", "mrr@10", "ndcg@10")


def _golden_sha() -> str:
    """골든셋 지문 = queries + qrels 동시 해시. 정답 청크(qrels)가 바뀌어도 SHA가 바뀌어야 재현 메타가 정확.

    각 파일을 이름·길이로 구분(경계 프리픽스)해 해시 → 단순 바이트 연결의 모호성(다른 분할이 같은
    연결열을 만들면 동일 SHA) 제거.
    """
    h = hashlib.sha256()
    for p in (_QUERIES, _QRELS):  # 고정 순서 + 경계(이름·길이) 정보
        data = p.read_bytes()
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()[:12]


def _qmetrics(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    """질의 1건의 검색 지표 (top_n까지 받은 ranked vs 정답 set)."""
    return {
        "hit@5": hit_rate_at_k(ranked, relevant, 5),
        "recall@5": recall_at_k(ranked, relevant, 5),
        "mrr@10": reciprocal_rank(ranked, relevant, 10),
        "ndcg@10": ndcg_at_k(ranked, relevant, 10),
    }


def _groups(g: GoldenQuery, pred: list[str]) -> list[str]:
    """질의가 속한 분석 그룹 태그. 공식=router_domains 카디널리티, 나머지는 보조."""
    gold = list(g.router_domains)
    tags = ["router_multi" if len(gold) >= 2 else "router_single"]
    tags.append("pred_has_secondary" if len(pred) >= 2 else "pred_single")
    tags.append("primary_correct" if pred and gold and pred[0] == gold[0] else "primary_wrong")
    if len(pred) >= 2:  # secondary 정합 — "라우터 문제 vs 문서 라벨 문제" 분리
        sec = pred[1]
        tags.append("secondary_in_router_gold" if sec in gold else "secondary_not_in_router_gold")
        tags.append("secondary_in_gold_domains" if sec in g.gold_domains else "secondary_not_in_gold_domains")
    return tags


def _group_metrics(rows: list[dict]) -> dict:
    """행 묶음의 A·B 평균 지표와 Δ(B−A). **raw float** — 성공기준은 raw로 판정하고 출력만 반올림(_round_floats)."""
    if not rows:
        return {"n": 0, "A": dict.fromkeys(_METRIC_KEYS), "B": dict.fromkeys(_METRIC_KEYS), "delta_B_minus_A": {}}
    a = {k: sum(r["a"][k] for r in rows) / len(rows) for k in _METRIC_KEYS}
    b = {k: sum(r["b"][k] for r in rows) / len(rows) for k in _METRIC_KEYS}
    delta = {k: b[k] - a[k] for k in _METRIC_KEYS}
    return {"n": len(rows), "A": a, "B": b, "delta_B_minus_A": delta}


def _round_floats(obj, ndigits: int = 4):
    """JSON 출력용: 모든 float 리프를 ndigits로 반올림(판정 후 적용 — 경계값 뒤집힘 방지)."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


_ALL_TAGS = (
    "router_multi",
    "router_single",
    "pred_has_secondary",
    "pred_single",
    "primary_correct",
    "primary_wrong",
    "secondary_in_router_gold",
    "secondary_not_in_router_gold",
    "secondary_in_gold_domains",
    "secondary_not_in_gold_domains",
)


def _per_metric_deltas(rows: list[dict]) -> dict:
    """**지표별**(hit@5·recall@5·mrr@10·ndcg@10) 질의별 Δ(B−A)와 improved/worsened 목록.

    Recall@5만 보면 순위 지표(MRR/NDCG)의 질의별 악화가 가려진다 → 전 지표를 각각 집계한다.
    """
    out: dict = {}
    for m in _METRIC_KEYS:
        diffs = [{"qid": r["qid"], "delta": round(r["b"][m] - r["a"][m], 4)} for r in rows]
        out[m] = {
            "improved": sorted([d for d in diffs if d["delta"] > 0], key=lambda d: -d["delta"]),
            "worsened": sorted([d for d in diffs if d["delta"] < 0], key=lambda d: d["delta"]),
        }
    return out


def _aggregate(rows: list[dict]) -> dict:
    """전체 + 그룹별 A/B/Δ + **지표별** 질의 개선·악화(전 지표)."""
    by_group = {tag: _group_metrics([r for r in rows if tag in r["groups"]]) for tag in _ALL_TAGS}
    return {
        "overall": _group_metrics(rows),
        "by_group": {t: m for t, m in by_group.items() if m["n"] > 0},
        "deltas_by_metric": _per_metric_deltas(rows),
    }


def _success_criteria(agg: dict) -> dict:
    """#86 §4.5 사전 성공기준 5개 판정 (B vs A).

    판정은 **raw float**(반올림 전 평균/Δ)로 하고, 표시값만 4자리로 반올림한다 → 경계값에서 통과/실패 안 뒤집힘.
    """
    ov = agg["overall"]
    multi = agg["by_group"].get("router_multi")
    single = agg["by_group"].get("router_single")
    # 과다예측 노이즈 코호트 = 라우터가 router_domains 정답에 **없는** secondary를 더한 질의(spurious).
    noise = agg["by_group"].get("secondary_not_in_router_gold")

    def d(group, key):
        return group["delta_B_minus_A"][key] if group and group["n"] else None

    def crit(passed: bool, raw):  # [통과여부, 표시용 Δ(반올림)]
        return [passed, round(raw, 4) if raw is not None else None]

    def crit_cohort(raw, ok):  # 코호트 0건 → "N/A"(통과로 위장 X), 있으면 [통과여부, Δ]
        return ["N/A", None] if raw is None else [ok(raw), round(raw, 4)]

    overall_recall_d = d(ov, "recall@5")
    overall_ndcg_d = d(ov, "ndcg@10")
    multi_recall_d = d(multi, "recall@5")
    single_recall_d = d(single, "recall@5")
    noise_recall_d = d(noise, "recall@5")
    noise_ndcg_d = d(noise, "ndcg@10")
    return {
        "overall_recall@5_no_drop": crit(overall_recall_d is not None and overall_recall_d >= 0, overall_recall_d),
        "overall_ndcg@10_drop_within_2pp": crit(overall_ndcg_d is not None and overall_ndcg_d >= -0.02, overall_ndcg_d),
        "multi_recall@5_improved": crit(multi_recall_d is not None and multi_recall_d > 0, multi_recall_d),
        "single_recall@5_no_worsen": crit(single_recall_d is None or single_recall_d >= 0, single_recall_d),
        # #5 secondary 과다예측 노이즈 제한: spurious secondary 코호트(secondary∉router_domains)에서
        #    Recall@5 무하락 + NDCG@10 하락 ≤2%p. 코호트 0건이면 N/A(통과 아님).
        "secondary_overpred_noise_recall@5_no_drop": crit_cohort(noise_recall_d, lambda x: x >= 0),
        "secondary_overpred_noise_ndcg@10_within_2pp": crit_cohort(noise_ndcg_d, lambda x: x >= -0.02),
        "_note": "노이즈 코호트=secondary_not_in_router_gold(라우터 과다예측). 추가 진단은 by_group.secondary_not_in_*의 Δ.",
    }


def _corpus_fingerprint(settings) -> dict:
    """색인 코퍼스 지문(컬렉션명·청크수) — 결과가 어느 코퍼스에서 측정됐는지 추적(코퍼스가 변수)."""
    from app.db.qdrant import get_qdrant

    try:
        info = get_qdrant().get_collection(settings.qdrant_collection)
        return {"collection": settings.qdrant_collection, "points_count": info.points_count}
    except Exception:
        return {"collection": settings.qdrant_collection, "points_count": None}


def _missing_predictions(answerable: list[GoldenQuery], cache: dict[str, dict], meta) -> list[str]:
    """예측 캐시가 없거나 stale한 answerable qid (A/B는 전체 신선해야 공정)."""
    return sorted(
        g.qid
        for g in answerable
        if not (g.qid in cache and is_fresh(cache[g.qid], query_sha=sha12(g.query), meta=meta))
    )


async def _run(cache_path: str) -> dict:
    settings = get_settings()
    golden = load_golden_set(_QUERIES, _QRELS)
    answerable = [g for g in golden if g.is_answerable and g.relevant_chunk_ids]
    cache = load_cache(cache_path)
    meta = current_meta("", settings)

    missing = _missing_predictions(answerable, cache, meta)
    if missing:
        sys.exit(
            f"✗ 예측 캐시 누락/stale {len(missing)}건 {missing[:10]} — A/B는 전체 신선 캐시가 필요합니다. "
            f"먼저 'python scripts/eval_router_domains.py --build-cache'."
        )

    rows: list[dict] = []
    total = len(answerable)
    print(f"A/B 측정 시작: {total}문항 (질문당 검색·리랭크 1회, 리랭커 CPU라 ~10초/문항)", file=sys.stderr, flush=True)
    for i, g in enumerate(answerable, 1):
        print(f"[{i}/{total}] {g.qid}", file=sys.stderr, flush=True)
        pred = list(cache[g.qid].get("predicted_domains") or [])
        base = await base_soft_candidates(g.query, top_k=TOP_K, settings=settings)
        relevant = set(g.relevant_chunk_ids)
        a_ids = rank_chunk_ids_from_base(base, pred[:1], settings, TOP_N)
        b_ids = rank_chunk_ids_from_base(base, pred[:2], settings, TOP_N)
        rows.append(
            {
                "qid": g.qid,
                "pred": pred,
                "groups": _groups(g, pred),
                "a": _qmetrics(a_ids, relevant),
                "b": _qmetrics(b_ids, relevant),
            }
        )

    agg = _aggregate(rows)  # raw float
    criteria = _success_criteria(agg)  # 판정은 raw로
    agg_out = _round_floats(agg)  # JSON 출력만 반올림
    return {
        "eval_datetime": datetime.now(UTC).isoformat(),
        "arms": {"A": "predicted_domains[:1]", "B": "predicted_domains[:2]"},
        "reproduction": {
            "golden_sha": _golden_sha(),
            "code_commit_sha": git_commit_sha(),
            "requested_model": meta.requested_model,
            "effective_provider": meta.effective_provider,
            "llm_provider": meta.llm_provider,
            "default_model": meta.default_model,
            "prompt_sha": meta.prompt_sha,
            "schema_version": meta.schema_version,
            "top_k": TOP_K,
            "top_n": TOP_N,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "filter_mode": "soft",
            "corpus": _corpus_fingerprint(settings),
            "search_query_source": "golden_query_raw",  # ⚠ 운영 라우터의 refined_query 아님 — 아래 scope 참조
            "note": "A/B는 동일 예측 캐시·동일 base(검색·리랭크 1회)로 secondary 도메인 가산만 격리. baseline.json(=g.domain 회귀게이트)과 다름.",
            "scope": (
                "검색에 골든 질의 원문(g.query)을 사용 — secondary 도메인 가산 효과만 격리한 유효 실험이나, "
                "운영은 라우터의 refined_query로 검색하므로 '운영 라우터 전체 효과'가 아니다. "
                "운영 경로 A/B는 캐시에 refined_query를 저장해 별도 측정(후속)."
            ),
        },
        "counts": {
            "answerable_evaluated": len(rows),
            "router_multi": sum(1 for r in rows if "router_multi" in r["groups"]),
            "router_single": sum(1 for r in rows if "router_single" in r["groups"]),
            "pred_with_secondary": sum(1 for r in rows if "pred_has_secondary" in r["groups"]),
        },
        "success_criteria_B_vs_A": criteria,
        **agg_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="검색 A/B (질의 단일 vs 멀티 도메인 가산)")
    parser.add_argument("--cache", default=_DEFAULT_CACHE, help="라우터 예측 캐시 경로")
    parser.add_argument(
        "--write-result", nargs="?", const=_DEFAULT_RESULT, default=None, help=f"결과 JSON 저장(기본 {_DEFAULT_RESULT})"
    )
    args = parser.parse_args()

    result = asyncio.run(_run(args.cache))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.write_result:
        d = os.path.dirname(args.write_result)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = args.write_result + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        os.replace(tmp, args.write_result)
        print(f"\n✅ A/B 결과 저장: {args.write_result}")


if __name__ == "__main__":
    main()
