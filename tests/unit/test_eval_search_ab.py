"""검색 A/B 하니스 단위 테스트 (Qdrant 불필요 — paired 가산·그룹·집계·성공기준·캐시 가드)."""

from app.config import Settings
from app.eval.dataset import GoldenQuery
from app.eval.retrieval_adapter import rank_chunk_ids_from_base
from app.eval.router_cache import CacheMeta
from scripts.eval_search_ab import (
    _aggregate,
    _groups,
    _missing_predictions,
    _qmetrics,
    _success_criteria,
)


def _g(qid, router, gold=()):
    return GoldenQuery(
        qid, "q", None, True, ("x",), gold_domains=gold, router_domains=router, router_domains_source="explicit"
    )


# ── paired 가산: 같은 base에 도메인만 달리 적용 → secondary 사용 여부만 차이 ──
def test_paired_gain_promotes_secondary_doc():
    s = Settings()  # primary 0.1 > secondary 0.05
    base = [(0.50, {"chunk_id": "sec", "domain": "manual"}), (0.52, {"chunk_id": "other", "domain": "planning"})]
    # A=[incident]: 둘 다 무가산 → vector 순 [other(0.52), sec(0.50)]
    assert rank_chunk_ids_from_base(base, ["incident"], s, 2) == ["other", "sec"]
    # B=[incident, manual]: sec=manual=secondary(+0.05)=0.55 > other(0.52) → 순서 뒤집힘
    assert rank_chunk_ids_from_base(base, ["incident", "manual"], s, 2) == ["sec", "other"]


# ── 그룹 태그 ──
def test_groups_multi_with_correct_secondary():
    g = _g("m1", ("incident", "manual"), ("incident", "manual"))
    tags = _groups(g, ["incident", "manual"])
    assert {
        "router_multi",
        "pred_has_secondary",
        "primary_correct",
        "secondary_in_router_gold",
        "secondary_in_gold_domains",
    } <= set(tags)


def test_groups_single_with_spurious_secondary():
    g = _g("d1", ("api_reference",), ("api_reference",))
    tags = _groups(g, ["api_reference", "manual"])  # 라우터가 없는 secondary를 더함
    assert {
        "router_single",
        "pred_has_secondary",
        "primary_correct",
        "secondary_not_in_router_gold",
        "secondary_not_in_gold_domains",
    } <= set(tags)


# ── 질의 지표 ──
def test_qmetrics_perfect_and_miss():
    assert _qmetrics(["c1", "c2"], {"c1"})["recall@5"] == 1.0
    assert _qmetrics(["x", "y"], {"c1"})["hit@5"] == 0.0


# ── 집계 + 성공기준 ──
def _m(h, r, mr, nd):
    return {"hit@5": h, "recall@5": r, "mrr@10": mr, "ndcg@10": nd}


def _row(qid, groups, a, b):
    return {"qid": qid, "pred": [], "groups": groups, "a": a, "b": b}


def test_aggregate_and_success_criteria():
    rows = [
        _row("m1", ["router_multi", "pred_has_secondary"], _m(1, 0.5, 0.5, 0.5), _m(1, 0.8, 0.6, 0.6)),  # B 개선
        _row("d1", ["router_single", "pred_single"], _m(1, 0.6, 0.6, 0.6), _m(1, 0.6, 0.6, 0.6)),  # 동일
    ]
    agg = _aggregate(rows)
    assert agg["overall"]["delta_B_minus_A"]["recall@5"] > 0
    rec = agg["deltas_by_metric"]["recall@5"]
    assert any(d["qid"] == "m1" for d in rec["improved"]) and rec["worsened"] == []
    # 순위 지표도 질의별로 잡힌다(B가 m1의 mrr/ndcg 개선)
    assert any(d["qid"] == "m1" for d in agg["deltas_by_metric"]["mrr@10"]["improved"])
    sc = _success_criteria(agg)
    assert sc["overall_recall@5_no_drop"][0] is True
    assert sc["multi_recall@5_improved"][0] is True
    assert sc["single_recall@5_no_worsen"][0] is True


def test_success_criteria_flags_multi_no_improvement():
    rows = [_row("m1", ["router_multi"], _m(1, 0.6, 0.6, 0.6), _m(1, 0.6, 0.6, 0.6))]  # 멀티 개선 없음
    sc = _success_criteria(_aggregate(rows))
    assert sc["multi_recall@5_improved"][0] is False


# ── 캐시 완전성 가드 ──
def test_missing_predictions_detects_empty_cache():
    meta = CacheMeta("", "openai", "", "gpt-4o-mini", "abc", "1")
    assert _missing_predictions([_g("d1", ("manual",))], {}, meta) == ["d1"]
