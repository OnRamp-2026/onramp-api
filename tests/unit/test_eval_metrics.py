"""결정론 메트릭 단위 테스트 (의존성 0)."""

import math

import pytest

from app.eval.metrics import (
    aggregate,
    answerability_accuracy,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["a", "b", "c"]
REL = {"b"}


def test_hit_rate() -> None:
    assert hit_rate_at_k(RANKED, REL, 1) == 0.0  # a 만 → 미포함
    assert hit_rate_at_k(RANKED, REL, 2) == 1.0  # a,b → 포함
    assert hit_rate_at_k(RANKED, REL, 3) == 1.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(RANKED, REL, 3) == 0.5  # b 가 2위 → 1/2
    assert reciprocal_rank(RANKED, REL, 1) == 0.0
    assert reciprocal_rank(RANKED, {"z"}, 3) == 0.0


def test_recall() -> None:
    assert recall_at_k(RANKED, {"b", "c"}, 3) == 1.0
    assert recall_at_k(RANKED, {"b", "c"}, 2) == 0.5  # top2=a,b → b만
    assert recall_at_k(RANKED, {"z"}, 3) == 0.0


def test_ndcg_binary() -> None:
    # 정답 b 가 2위 → DCG = 1/log2(3), IDCG = 1/log2(2) = 1
    expected = (1.0 / math.log2(3)) / 1.0
    assert ndcg_at_k(RANKED, REL, 3) == pytest.approx(expected)
    # 정답이 1위면 nDCG = 1
    assert ndcg_at_k(["b", "a"], REL, 2) == pytest.approx(1.0)


def test_empty_relevant_returns_zero() -> None:
    assert hit_rate_at_k(RANKED, set(), 3) == 0.0
    assert reciprocal_rank(RANKED, set(), 3) == 0.0
    assert recall_at_k(RANKED, set(), 3) == 0.0
    assert ndcg_at_k(RANKED, set(), 3) == 0.0


def test_empty_ranked() -> None:
    assert hit_rate_at_k([], REL, 5) == 0.0
    assert recall_at_k([], REL, 5) == 0.0


def test_k_larger_than_ranked() -> None:
    assert hit_rate_at_k(RANKED, REL, 100) == 1.0


def test_aggregate_excludes_empty_relevant() -> None:
    per_query = [
        (["b", "a", "c"], {"b"}),  # hit@5=1, rr=1.0, recall@5=1
        (["x", "y"], set()),  # unanswerable → 제외
    ]
    summary = aggregate(per_query, k_hit=5, k_mrr=10, k_recall=5, k_ndcg=10)
    assert summary.n == 1  # 빈 relevant 제외
    assert summary.hit_rate == 1.0
    assert summary.mrr == 1.0
    assert summary.recall == 1.0
    assert summary.as_dict()["hit_rate@5"] == 1.0


def test_aggregate_all_empty() -> None:
    summary = aggregate([([], set())])
    assert summary.n == 0
    assert summary.hit_rate == 0.0


def test_answerability_accuracy() -> None:
    preds = [True, False, True, False]
    labels = [True, False, False, True]
    s = answerability_accuracy(preds, labels)
    assert (s.tp, s.fp, s.tn, s.fn) == (1, 1, 1, 1)
    assert s.accuracy == 0.5
    assert s.precision == 0.5
    assert s.recall == 0.5
    assert s.f1 == pytest.approx(0.5)


def test_answerability_perfect() -> None:
    s = answerability_accuracy([True, False], [True, False])
    assert s.accuracy == 1.0
    assert s.f1 == 1.0


def test_answerability_length_mismatch() -> None:
    with pytest.raises(ValueError, match="길이"):
        answerability_accuracy([True], [True, False])
