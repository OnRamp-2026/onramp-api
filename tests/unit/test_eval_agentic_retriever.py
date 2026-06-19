from scripts.eval_agentic_retriever import (
    _aggregate_strategy,
    _build_report,
    _markdown_report,
    _percentile,
)


def _row(
    strategy: str,
    *,
    qid: str = "q1",
    latency_ms: float = 100.0,
    ranked: list[str] | None = None,
    relevant: list[str] | None = None,
    calls: list[dict] | None = None,
    rrf: bool = False,
    fallback: str | None = None,
) -> dict:
    return {
        "strategy": strategy,
        "qid": qid,
        "repeat": 1,
        "latency_ms": latency_ms,
        "ranked_chunk_ids": ranked or ["c1", "c2"],
        "relevant_chunk_ids": relevant or ["c1"],
        "diagnostics": {
            "calls": calls or [],
            "tool_call_count": len(calls or []),
            "duplicate_calls": 0,
            "ranking_list_count": len(calls or []),
            "rrf_applied": rrf,
            "fallback": fallback,
            "rerank_fallback": False,
        },
    }


def test_percentile_interpolates_small_samples() -> None:
    assert _percentile([100.0, 200.0, 300.0], 0.5) == 200.0
    assert _percentile([100.0, 200.0, 300.0], 0.95) == 290.0


def test_aggregate_strategy_reports_quality_latency_and_agentic_diagnostics() -> None:
    rows = [
        _row(
            "agentic",
            latency_ms=100.0,
            calls=[{"tool": "search_dense"}, {"tool": "search_bm25"}],
            rrf=True,
        ),
        _row(
            "agentic",
            qid="q2",
            latency_ms=300.0,
            ranked=["x"],
            relevant=["z"],
            calls=[{"tool": "search_dense"}],
            fallback="no_search_results",
        ),
    ]

    summary = _aggregate_strategy(rows)

    assert summary["quality"]["hit_rate@5"] == 0.5
    assert summary["quality"]["mrr@10"] == 0.5
    assert summary["latency_ms"] == {"avg": 200.0, "p50": 200.0, "p95": 290.0}
    assert summary["agentic"]["avg_tool_calls"] == 1.5
    assert summary["agentic"]["dense_selection_ratio"] == 1.0
    assert summary["agentic"]["bm25_selection_ratio"] == 0.5
    assert summary["agentic"]["multi_tool_ratio"] == 0.5
    assert summary["agentic"]["rrf_ratio"] == 0.5
    assert summary["agentic"]["fallback_ratio"] == 0.5


def test_report_contains_delta_and_markdown_table() -> None:
    rows = [
        _row("deterministic", latency_ms=100.0),
        _row(
            "agentic",
            latency_ms=250.0,
            calls=[{"tool": "search_dense"}],
        ),
    ]

    report = _build_report(rows, repeats=1, model="gpt-4o-mini", tenant_id="tenant1")
    markdown = _markdown_report(report)

    assert report["delta"]["hit_rate@5"] == 0.0
    assert report["delta"]["avg_latency_ms"] == 150.0
    assert "| Hit Rate@5 | 1.0000 | 1.0000 | +0.0000 |" in markdown
    assert "| 평균 지연(ms) | 100.00 | 250.00 | +150.00 |" in markdown
    assert "Agentic 실행 진단" in markdown
