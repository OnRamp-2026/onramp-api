import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "eval_single_agentic_ab.py"
    spec = importlib.util.spec_from_file_location("eval_single_agentic_ab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_contains_quality_latency_tools_retries_fallback_and_tokens():
    summary = _module().summarize(
        [
            {
                "ranked_chunk_ids": ["c1"],
                "relevant_chunk_ids": ["c1"],
                "latency_ms": 100,
                "tool_trace": [{"tool": "hybrid_search", "fallback": ""}],
                "retry_count": 1,
                "tokens": 20,
            }
        ]
    )

    assert summary["quality"]["recall@5"] == 1.0
    assert summary["latency_ms"] == {"avg": 100.0, "p50": 100.0, "p95": 100.0}
    assert summary["avg_tool_calls"] == 1.0
    assert summary["retry_ratio"] == 1.0
    assert summary["fallback_ratio"] == 0.0
    assert summary["avg_tokens"] == 20.0
