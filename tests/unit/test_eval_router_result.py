"""eval_router_domains 결과 빌더 단위 테스트 (near-miss 분리 + explicit 게이트)."""

from app.eval.dataset import GoldenQuery
from scripts.eval_router_domains import _block_breakdown, _metrics_blocks


def _gq(qid, answerable, router=()):
    src = "explicit" if router else "none"
    return GoldenQuery(qid, "q", None, answerable, (), router_domains=router, router_domains_source=src)


def test_block_breakdown_splits_near_miss_and_out_of_scope():
    golden = [_gq("n001", False), _gq("n002", False), _gq("d050", False), _gq("d001", True)]
    by_qid = {
        "n001": {"use_case": "답변불가"},  # near-miss 차단됨
        "n002": {"use_case": "검색"},  # near-miss 통과(미차단)
        "d050": {"use_case": "답변불가"},  # 사외 차단됨
        "d001": {"use_case": "검색"},  # answerable 정상
    }
    b = _block_breakdown(golden, by_qid)
    assert b["near_miss_n0xx"]["blocked"] == 1 and b["near_miss_n0xx"]["n"] == 2 and b["near_miss_n0xx"]["rate"] == 0.5
    assert b["out_of_scope"]["blocked"] == 1 and b["out_of_scope"]["n"] == 1 and b["out_of_scope"]["rate"] == 1.0
    assert b["total"]["blocked"] == 2 and b["total"]["n"] == 3
    assert b["answerable_false_block"]["false_blocked"] == 0 and b["answerable_false_block"]["n"] == 1


def test_block_breakdown_zero_denominator_is_none():
    b = _block_breakdown([_gq("d001", True)], {"d001": {"use_case": "검색"}})
    assert b["near_miss_n0xx"]["rate"] is None  # near-miss 0건 → N/A
    assert b["out_of_scope"]["rate"] is None
    assert b["total"]["rate"] is None


def test_metrics_blocks_none_when_no_explicit():
    # fallback(미검수) 질문만 있으면 공식 지표 없음 → None (baseline도 안 써짐)
    golden = [_gq("d001", True)]  # router_domains 없음 → explicit 아님
    assert (
        _metrics_blocks(golden, {"d001": {"predicted_domains": ["manual"], "raw_predicted_domains": ["manual"]}})
        is None
    )


def test_metrics_blocks_present_with_explicit():
    golden = [_gq("m001", True, ("incident", "manual"))]
    by_qid = {
        "m001": {
            "predicted_domains": ["incident", "manual"],
            "raw_predicted_domains": ["incident", "manual"],
            "confidence": 0.9,
            "parse_ok": True,
            "low_conf_empty": False,
        }
    }
    res = _metrics_blocks(golden, by_qid)
    assert res is not None
    raw_m, eff_d = res
    assert raw_m.n_eval == 1 and raw_m.primary_accuracy == 1.0
    assert "ece" not in eff_d  # effective 블록은 calibration 제거
