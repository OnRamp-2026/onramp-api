"""에이전트 운영지표 — 인프로세스 카운터 + Prometheus 텍스트 렌더.

`prometheus_client` 의존성 없이 레포의 수동 렌더 패턴(prometheus_metrics.render_worker_metrics)과
동일하게, single_agentic 경로의 동작을 시계열 모니터링용으로 `/metrics`에 노출한다.
카운터는 프로세스-로컬(재시작 시 리셋) — Prometheus scrape 모델에 부합(monotonic counter).

#273(요청 1건 Langfuse trace=디버깅)과 보완: 여기는 **집계 추세/알림용**(예: fallback율 급증).
"""

from __future__ import annotations

import threading
from collections import defaultdict

_lock = threading.Lock()
_tool_calls: dict[str, int] = defaultdict(int)  # 도구별 호출 수
_fallbacks = 0  # tool 실행 fallback(빈결과·예외) 발생 수
_retry_steps = 0  # 재검색(retry) 스텝 수
_steps = 0  # 에이전틱 검색 스텝 총 수


def record_agentic_step(tools: list[str], *, fallbacks: int, retried: bool) -> None:
    """한 single_agentic 검색 스텝의 도구 호출/실패/재시도를 카운트한다 (스레드 안전)."""
    global _fallbacks, _retry_steps, _steps
    with _lock:
        _steps += 1
        for tool in tools:
            _tool_calls[tool] += 1
        _fallbacks += fallbacks
        if retried:
            _retry_steps += 1


def reset() -> None:
    """테스트용 — 카운터 초기화."""
    global _fallbacks, _retry_steps, _steps
    with _lock:
        _tool_calls.clear()
        _fallbacks = _retry_steps = _steps = 0


def render() -> str:
    """현재 카운터를 Prometheus 텍스트 노출 형식으로 렌더한다."""
    with _lock:
        lines = [
            "# HELP onramp_agent_steps_total single_agentic 검색 스텝 총 수",
            "# TYPE onramp_agent_steps_total counter",
            f"onramp_agent_steps_total {_steps}",
            "# HELP onramp_agent_tool_calls_total 도구별 호출 수",
            "# TYPE onramp_agent_tool_calls_total counter",
        ]
        for tool, count in sorted(_tool_calls.items()):
            lines.append(f'onramp_agent_tool_calls_total{{tool="{tool}"}} {count}')
        lines += [
            "# HELP onramp_agent_tool_fallbacks_total 도구 실행 fallback 발생 수",
            "# TYPE onramp_agent_tool_fallbacks_total counter",
            f"onramp_agent_tool_fallbacks_total {_fallbacks}",
            "# HELP onramp_agent_retry_steps_total 재검색(retry) 스텝 수",
            "# TYPE onramp_agent_retry_steps_total counter",
            f"onramp_agent_retry_steps_total {_retry_steps}",
        ]
    return "\n".join(lines) + "\n"
