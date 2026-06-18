"""결정론 메트릭 단위 테스트 (의존성 0)."""

import math

import pytest

from app.eval.metrics import (
    aggregate,
    answerability_accuracy,
    chunk_to_page,
    collapse_to_pages,
    evidence_span_hit,
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


@pytest.mark.parametrize("fn", [hit_rate_at_k, reciprocal_rank, recall_at_k, ndcg_at_k])
@pytest.mark.parametrize("bad_k", [0, -1])
def test_non_positive_k_raises(fn, bad_k) -> None:
    with pytest.raises(ValueError, match="1 이상"):
        fn(RANKED, REL, bad_k)


def test_aggregate_non_positive_k_raises() -> None:
    with pytest.raises(ValueError, match="k_hit"):
        aggregate([(RANKED, REL)], k_hit=0)


# ── #212 page-level (splitter-독립) ──
def test_chunk_to_page_strips_index_suffix() -> None:
    assert chunk_to_page("107194_004") == "107194"  # 표준 포맷
    assert chunk_to_page("gh_repo_42_003") == "gh_repo_42"  # page_id에 _ 있어도 숫자 suffix만 제거
    assert chunk_to_page("noindex") == "noindex"  # suffix 없으면 원본
    assert chunk_to_page("name_abc") == "name_abc"  # 숫자 아닌 suffix는 page_id의 일부


def test_collapse_to_pages_dedupes_preserving_order() -> None:
    # 같은 page의 여러 chunk → 첫 등장 순위 하나로. 순서 보존.
    ranked = ["107194_004", "107194_001", "200_000", "107194_009", "300_002"]
    assert collapse_to_pages(ranked) == ["107194", "200", "300"]


def test_collapse_to_pages_reused_by_rank_metrics() -> None:
    # collapse 결과를 기존 chunk 지표 함수에 그대로 넣어 page-level 점수를 낸다(공정 비교 핵심).
    pages = collapse_to_pages(["107194_004", "107194_001", "200_000"])  # → ["107194","200"]
    assert hit_rate_at_k(pages, {"200"}, 2) == 1.0
    assert recall_at_k(pages, {"107194", "200"}, 5) == 1.0
    assert reciprocal_rank(pages, {"200"}, 5) == 0.5  # 200이 2위


def test_evidence_span_hit_normalizes_whitespace_and_case() -> None:
    contexts = ["...설정값 ID => 66 입니다...", "다른 문맥"]
    assert evidence_span_hit(contexts, "id => 66") == 1.0  # 대소문자·공백 무관
    assert evidence_span_hit(contexts, "id=>66") == 1.0
    assert evidence_span_hit(contexts, "id => 99") == 0.0  # 미포함
    assert evidence_span_hit(contexts, "") == 0.0  # span 없으면 대상 아님
    assert evidence_span_hit(["맞는 답 id => 66"], "id => 66", k=0) == 0.0  # top-0이면 빈 풀
