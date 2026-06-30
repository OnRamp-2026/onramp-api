"""에이전틱 평가 하네스 지표 — 순수 함수(그래프 실행과 분리해 단위 테스트 가능).

행(row)은 `scripts/eval_agentic.py`가 그래프 1회 실행으로 만든 dict:
    {qid, ranked_chunk_ids, relevant_chunk_ids, answerability,
     tool_sources(list[str]), expected_source(str|None),
     any_fallback(bool), retry_count(int), latency_ms(float), tokens(int)}
"""

from __future__ import annotations

from typing import Any

from app.eval.metrics import aggregate


def expected_source(page_ids: list[str] | tuple[str, ...]) -> str | None:
    """골든 page_ids에서 기대 source를 결정론으로 도출한다 (혼합/불명이면 None).

    `gh:owner/repo#n` 형식 → github, 그 외(Confluence 숫자 page id) → confluence.
    한 질의의 정답 근거가 한 source로 일관될 때만 평가 대상으로 삼는다.
    """
    if not page_ids:
        return None
    sources = {"github" if str(pid).startswith("gh:") else "confluence" for pid in page_ids}
    if len(sources) != 1:
        return None  # 혼합 source 질의는 tool-selection 평가에서 제외
    return sources.pop()


def tool_selection_stats(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """source-filtered 검색의 라우팅 정확도 — **초기 source 선택**만 평가한다.

    이후 재시도로 올바른 source를 뒤늦게 시도한 것(recovery)은 초기 라우팅 정확도와
    별개이므로 첫 source-filtered 호출만 채점한다(set 매칭이 아니라 순서상 첫 선택).
    - correct: 첫 source 선택이 기대 source와 일치
    - misrouted: 첫 source 선택이 다른 source — github 질의에 confluence 등
    - neutral: source 미지정(무제한 hybrid만) — 오답 아님, 정확도 분모 제외
    accuracy = correct / (correct + misrouted). 평가 가능 행이 없으면 None.
    """
    correct = misrouted = neutral = 0
    for row in rows:
        exp = row.get("expected_source")
        if not exp:
            continue
        sources = row.get("tool_sources") or []
        first = sources[0] if sources else None  # 초기 선택만 평가 (recovery 제외)
        if first is None:  # source 미지정(무제한 hybrid) → 중립
            neutral += 1
        elif first == exp:
            correct += 1
        else:  # 첫 선택이 다른 source → 오라우팅
            misrouted += 1
    evaluable = correct + misrouted
    if evaluable == 0 and neutral == 0:
        return None
    return {
        "accuracy": round(correct / evaluable, 4) if evaluable else None,
        "correct": correct,
        "misrouted": misrouted,
        "neutral": neutral,
        "n_evaluable": evaluable,
    }


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    import math

    idx = max(0, math.ceil(len(ordered) * ratio) - 1)
    return round(ordered[idx], 2)


def summarize_arm(arm: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """한 arm의 행들을 지표로 집계한다 (검색·answerability·운영·tool-selection)."""
    n = len(rows)
    # 검색 지표 (qrels 기준, app/eval 재사용) — relevant 빈셋 질문은 aggregate가 자동 제외
    retrieval = aggregate(
        [(list(r.get("ranked_chunk_ids", [])), set(r.get("relevant_chunk_ids", []))) for r in rows]
    ).as_dict()
    answerable = sum(1 for r in rows if r.get("answerability") in {"answerable", "partially_answerable"})
    fallbacks = sum(1 for r in rows if r.get("any_fallback"))
    retries = sum(1 for r in rows if int(r.get("retry_count", 0)) > 0)
    latencies = [float(r.get("latency_ms", 0.0)) for r in rows]
    tokens = [int(r.get("tokens", 0)) for r in rows]
    return {
        "arm": arm,
        "n": n,
        "retrieval": retrieval,
        "answerable_or_partial_rate": round(answerable / n, 4) if n else 0.0,
        "tool_selection": tool_selection_stats(rows),
        "fallback_rate": round(fallbacks / n, 4) if n else 0.0,
        "retry_rate": round(retries / n, 4) if n else 0.0,
        "latency_p50_ms": _percentile(latencies, 0.5),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "mean_tokens": round(sum(tokens) / n) if n else 0,
    }


def _flat_metrics(summary: dict[str, Any]) -> dict[str, float | None]:
    """arm 요약에서 반복 집계 대상 헤드라인 수치만 평탄화."""
    r = summary.get("retrieval", {}) or {}
    ts = summary.get("tool_selection") or {}
    return {
        "hit_rate@5": r.get("hit_rate@5"),
        "recall@5": r.get("recall@5"),
        "mrr@10": r.get("mrr@10"),
        "ndcg@10": r.get("ndcg@10"),
        "answerable_or_partial_rate": summary.get("answerable_or_partial_rate"),
        "tool_selection_accuracy": ts.get("accuracy"),
        "fallback_rate": summary.get("fallback_rate"),
        "retry_rate": summary.get("retry_rate"),
        "latency_p50_ms": summary.get("latency_p50_ms"),
        "mean_tokens": summary.get("mean_tokens"),
    }


def aggregate_repeats(arm: str, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """반복 실행한 arm 요약들을 지표별 평균±표준편차로 집계한다.

    LLM 확률성을 통제해 arm 간 차이가 노이즈인지 실제인지(std로) 보이게 한다.
    값이 None인 지표(예: deterministic의 tool_selection)는 평균에서 제외한다.
    """
    import statistics

    flats = [_flat_metrics(s) for s in summaries]
    metrics: dict[str, Any] = {}
    keys = flats[0].keys() if flats else []
    for k in keys:
        vals: list[float] = [float(v) for f in flats if isinstance((v := f.get(k)), int | float)]
        if not vals:
            metrics[k] = None
            continue
        metrics[k] = {
            "mean": round(statistics.mean(vals), 4),
            "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return {"arm": arm, "repeats": len(summaries), "metrics": metrics}
