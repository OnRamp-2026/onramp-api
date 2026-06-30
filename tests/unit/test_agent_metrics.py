"""에이전트 운영지표 단위 테스트 (인프로세스 카운터 + Prometheus 렌더)."""

from app.observability import agent_metrics


def setup_function() -> None:
    agent_metrics.reset()


def test_record_and_render_counters():
    agent_metrics.record_agentic_step(["hybrid_search", "hybrid_search_by_source"], fallbacks=1, retried=False)
    agent_metrics.record_agentic_step(["hybrid_search"], fallbacks=0, retried=True)
    out = agent_metrics.render()
    assert "onramp_agent_steps_total 2" in out
    assert 'onramp_agent_tool_calls_total{tool="hybrid_search"} 2' in out
    assert 'onramp_agent_tool_calls_total{tool="hybrid_search_by_source"} 1' in out
    assert "onramp_agent_tool_fallbacks_total 1" in out
    assert "onramp_agent_retry_steps_total 1" in out


def test_render_prometheus_format_has_help_type():
    agent_metrics.record_agentic_step(["hybrid_search"], fallbacks=0, retried=False)
    out = agent_metrics.render()
    assert "# HELP onramp_agent_steps_total" in out
    assert "# TYPE onramp_agent_steps_total counter" in out
    assert out.endswith("\n")


def test_reset_clears():
    agent_metrics.record_agentic_step(["hybrid_search"], fallbacks=2, retried=True)
    agent_metrics.reset()
    out = agent_metrics.render()
    assert "onramp_agent_steps_total 0" in out
    assert "onramp_agent_tool_fallbacks_total 0" in out
    # 도구별 라인은 카운터가 비면 안 나온다
    assert "onramp_agent_tool_calls_total{" not in out
