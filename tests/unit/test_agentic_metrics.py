"""에이전틱 평가 하네스 지표 단위 테스트 (그래프 실행 없이 순수 함수 검증)."""

from app.eval.agentic_metrics import expected_source, summarize_arm, tool_selection_stats


def test_expected_source_from_page_ids():
    assert expected_source(["gh:gitops#31"]) == "github"
    assert expected_source(["174178"]) == "confluence"
    assert expected_source(["gh:a#1", "gh:b#2"]) == "github"
    assert expected_source(["gh:a#1", "174178"]) is None  # 혼합 source → 평가 제외
    assert expected_source([]) is None


def test_tool_selection_stats_correct_misrouted_neutral():
    rows = [
        {"expected_source": "github", "tool_sources": ["github"]},  # correct
        {"expected_source": "github", "tool_sources": ["confluence"]},  # misrouted
        {"expected_source": "confluence", "tool_sources": []},  # neutral (무제한 hybrid)
        {"expected_source": None, "tool_sources": ["github"]},  # 평가 대상 아님
    ]
    stats = tool_selection_stats(rows)
    assert stats == {"accuracy": 0.5, "correct": 1, "misrouted": 1, "neutral": 1, "n_evaluable": 2}


def test_tool_selection_stats_none_when_nothing_evaluable():
    assert tool_selection_stats([{"expected_source": None, "tool_sources": []}]) is None


def test_tool_selection_scores_first_pick_only():
    # 첫 선택이 오source → recovery로 뒤에 올바른 source를 시도해도 misrouted (초기 라우팅 평가)
    stats = tool_selection_stats([{"expected_source": "github", "tool_sources": ["confluence", "github"]}])
    assert stats["correct"] == 0 and stats["misrouted"] == 1
    # 첫 선택이 올바르면 correct
    stats2 = tool_selection_stats([{"expected_source": "github", "tool_sources": ["github", "confluence"]}])
    assert stats2["correct"] == 1 and stats2["misrouted"] == 0


def test_summarize_arm_aggregates_all_metrics():
    rows = [
        {
            "qid": "a",
            "ranked_chunk_ids": ["c1", "c9"],
            "relevant_chunk_ids": ["c1"],
            "answerability": "answerable",
            "tool_sources": ["github"],
            "expected_source": "github",
            "any_fallback": False,
            "retry_count": 0,
            "latency_ms": 100,
            "tokens": 50,
        },
        {
            "qid": "b",
            "ranked_chunk_ids": ["x"],
            "relevant_chunk_ids": ["c2"],
            "answerability": "not_enough_evidence",
            "tool_sources": ["confluence"],
            "expected_source": "github",
            "any_fallback": True,
            "retry_count": 2,
            "latency_ms": 300,
            "tokens": 150,
        },
    ]
    s = summarize_arm("single_agentic", rows)
    assert s["arm"] == "single_agentic"
    assert s["n"] == 2
    assert s["retrieval"]["hit_rate@5"] == 0.5  # a 적중, b 미적중
    assert s["answerable_or_partial_rate"] == 0.5
    assert s["tool_selection"]["accuracy"] == 0.5
    assert s["fallback_rate"] == 0.5
    assert s["retry_rate"] == 0.5
    assert s["latency_p50_ms"] == 100.0
    assert s["mean_tokens"] == 100


def test_summarize_arm_partial_counts_as_answerable():
    rows = [
        {
            "ranked_chunk_ids": [],
            "relevant_chunk_ids": [],
            "answerability": "partially_answerable",
            "tool_sources": [],
            "expected_source": None,
            "any_fallback": False,
            "retry_count": 0,
            "latency_ms": 10,
            "tokens": 1,
        },
    ]
    assert summarize_arm("x", rows)["answerable_or_partial_rate"] == 1.0


def test_aggregate_repeats_mean_std():
    from app.eval.agentic_metrics import aggregate_repeats

    s1 = {
        "retrieval": {"hit_rate@5": 0.8, "recall@5": 0.5, "mrr@10": 0.6, "ndcg@10": 0.6},
        "answerable_or_partial_rate": 0.9,
        "tool_selection": {"accuracy": 0.5},
        "fallback_rate": 0.0,
        "retry_rate": 0.1,
        "latency_p50_ms": 100,
        "mean_tokens": 50,
    }
    s2 = {
        "retrieval": {"hit_rate@5": 0.6, "recall@5": 0.5, "mrr@10": 0.6, "ndcg@10": 0.4},
        "answerable_or_partial_rate": 0.7,
        "tool_selection": {"accuracy": 0.7},
        "fallback_rate": 0.0,
        "retry_rate": 0.1,
        "latency_p50_ms": 200,
        "mean_tokens": 70,
    }
    agg = aggregate_repeats("single_agentic", [s1, s2])
    assert agg["arm"] == "single_agentic" and agg["repeats"] == 2
    assert agg["metrics"]["hit_rate@5"] == {"mean": 0.7, "std": 0.1, "n": 2}
    assert agg["metrics"]["tool_selection_accuracy"]["mean"] == 0.6
    assert agg["metrics"]["latency_p50_ms"]["mean"] == 150.0


def test_aggregate_repeats_skips_none_metrics():
    from app.eval.agentic_metrics import aggregate_repeats

    # deterministic: tool_selection None → 평균 대상에서 제외
    s = {
        "retrieval": {"hit_rate@5": 0.8, "recall@5": 0.5, "mrr@10": 0.6, "ndcg@10": 0.6},
        "answerable_or_partial_rate": 0.9,
        "tool_selection": None,
        "fallback_rate": 0.0,
        "retry_rate": 0.0,
        "latency_p50_ms": 100,
        "mean_tokens": 50,
    }
    agg = aggregate_repeats("deterministic", [s, s])
    assert agg["metrics"]["tool_selection_accuracy"] is None
    assert agg["metrics"]["hit_rate@5"]["std"] == 0.0  # 동일값 → std 0
