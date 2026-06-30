"""에이전틱 RAG 평가 자동화 하네스.

arms(전략·설정 변형) × 골든셋(도메인 필터 가능)을 그래프로 실행해
도구선택 정확도·검색(Hit@k/Recall/MRR/nDCG)·answerability·fallback/재시도·지연/토큰을
측정하고 비교 리포트를 낸다. 설계 결정(예: 원문 escalation)을 데이터로 판단하기 위한 도구.

예)
  # deterministic vs single_agentic, incident 도메인만
  python scripts/eval_agentic.py --domain incident --out report.json
  # 전체 도메인
  python scripts/eval_agentic.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings  # noqa: E402
from app.eval.agentic_metrics import aggregate_repeats, expected_source, summarize_arm  # noqa: E402
from app.eval.dataset import load_golden_set  # noqa: E402
from app.services.llm_selector import usage_accumulator  # noqa: E402

# arm = (label, retriever_strategy, env_overrides). env_overrides로 설정 변형(예: escalation) 추가 가능.
ARMS: list[tuple[str, str, dict[str, str]]] = [
    ("deterministic", "deterministic", {}),
    ("single_agentic", "single_agentic", {}),
]


def _answerability(result: dict[str, Any]) -> str:
    status = result.get("answerability_status")
    return getattr(status, "value", str(status or ""))


def _tool_sources(result: dict[str, Any]) -> list[str]:
    """source-filtered 검색에서 에이전트가 고른 source 목록 (무제한 hybrid는 제외)."""
    sources: list[str] = []
    for trace in result.get("tool_trace", []):
        tool = getattr(trace, "tool", None) if not isinstance(trace, dict) else trace.get("tool")
        src = getattr(trace, "source", None) if not isinstance(trace, dict) else trace.get("source")
        if tool == "hybrid_search_by_source" and src:
            sources.append(str(src))
    return sources


def _any_fallback(result: dict[str, Any]) -> bool:
    for trace in result.get("tool_trace", []):
        fb = getattr(trace, "fallback", None) if not isinstance(trace, dict) else trace.get("fallback")
        if fb:
            return True
    return False


async def run_arm(
    label: str, strategy: str, env_overrides: dict[str, str], golden: list, args: argparse.Namespace
) -> list[dict[str, Any]]:
    from app.agents.graph import compiled_graph  # import 시 graph 빌드 — arm 루프 밖 1회면 충분하나 명시 import

    saved_env = {k: os.environ.get(k) for k in env_overrides}  # 원복용 — arm 간 env 누수 방지
    for key, val in env_overrides.items():
        os.environ[key] = val
    get_settings.cache_clear()
    settings = get_settings()
    rows: list[dict[str, Any]] = []
    for item in golden:
        state = {
            "query": item.query,
            "model": args.model,
            "tenant_id": args.tenant_id or settings.auth_default_tenant,
            "retriever_strategy": strategy,
            "retry_count": 0,
            "max_retries": settings.trust_max_retries,
        }
        started = perf_counter()
        try:
            with usage_accumulator() as usage:
                result = await compiled_graph.ainvoke(state)
            tokens = usage["total"]
        except Exception as exc:  # noqa: BLE001
            rows.append({"qid": item.qid, "error": type(exc).__name__})
            continue
        rows.append(
            {
                "qid": item.qid,
                "ranked_chunk_ids": [d.chunk_id for d in result.get("documents", []) if d.chunk_id],
                "relevant_chunk_ids": list(item.relevant_chunk_ids),
                "answerability": _answerability(result),
                "tool_sources": _tool_sources(result),
                "expected_source": expected_source(item.page_ids),
                "any_fallback": _any_fallback(result),
                "retry_count": result.get("retry_count", 0),
                "latency_ms": round((perf_counter() - started) * 1000, 2),
                "tokens": tokens,
            }
        )
        print(f"  [{label}] {item.qid}: {rows[-1].get('answerability', 'ERR')}")
    # env_overrides 원복 — 다음 arm으로 누수 방지
    for key, prev in saved_env.items():
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev
    get_settings.cache_clear()
    return rows


def _print_table(summaries: list[dict[str, Any]]) -> None:
    print("\n=== SUMMARY ===")
    for s in summaries:
        ts = s.get("tool_selection")
        ts_str = f"acc={ts['accuracy']}({ts['correct']}/{ts['n_evaluable']}),neutral={ts['neutral']}" if ts else "n/a"
        r = s["retrieval"]
        print(
            f"[{s['arm']}] n={s['n']} | hit@5={r.get('hit_rate@5')} recall@5={r.get('recall@5')} "
            f"ndcg@10={r.get('ndcg@10')} | answerable={s['answerable_or_partial_rate']} | "
            f"tool_sel={ts_str} | fallback={s['fallback_rate']} retry={s['retry_rate']} | "
            f"p50={s['latency_p50_ms']}ms p95={s['latency_p95_ms']}ms tok={s['mean_tokens']}"
        )


def _print_aggregate(aggregates: list[dict[str, Any]]) -> None:
    print("\n=== AGGREGATE (mean±std over repeats) ===")
    keys = [
        "hit_rate@5",
        "recall@5",
        "ndcg@10",
        "answerable_or_partial_rate",
        "tool_selection_accuracy",
        "latency_p50_ms",
    ]
    for a in aggregates:
        m = a["metrics"]
        parts = []
        for k in keys:
            v = m.get(k)
            parts.append(f"{k}={v['mean']}±{v['std']}" if v else f"{k}=n/a")
        print(f"[{a['arm']}] repeats={a['repeats']} | " + " ".join(parts))


async def main() -> int:
    parser = argparse.ArgumentParser(description="에이전틱 RAG 평가 자동화 하네스")
    parser.add_argument("--queries", default="data/eval/queries.jsonl")
    parser.add_argument("--qrels", default="data/eval/qrels.jsonl")
    parser.add_argument("--domain", default=None, help="도메인 필터 (예: incident). 미지정 시 전체")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1, help="arm×질의 반복 실행 횟수 (LLM 확률성→평균±std)")
    parser.add_argument("--model", default="")
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    golden = [g for g in load_golden_set(args.queries, args.qrels) if g.is_answerable]
    if args.domain:
        golden = [g for g in golden if (g.domain or "") == args.domain]
    if args.limit:
        golden = golden[: args.limit]
    print(f"golden queries: {len(golden)} (domain={args.domain or 'all'}, repeats={args.repeats})")

    summaries: list[dict[str, Any]] = []  # 대표(첫 실행) — 표 호환
    aggregates: list[dict[str, Any]] = []  # 반복 평균±std
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for label, strategy, env in ARMS:
        per_repeat: list[dict[str, Any]] = []
        for rep in range(args.repeats):
            rows = await run_arm(label, strategy, env, golden, args)
            ok = [r for r in rows if "error" not in r]
            per_repeat.append({**summarize_arm(label, ok), "errors": len(rows) - len(ok)})
            all_rows[f"{label}#{rep + 1}"] = rows
        summaries.append(per_repeat[0])
        aggregates.append(aggregate_repeats(label, per_repeat))

    _print_table(summaries)
    if args.repeats > 1:
        _print_aggregate(aggregates)
    report = {
        "generated_at_note": "stamp externally",
        "config": {
            "domain": args.domain or "all",
            "queries": len(golden),
            "repeats": args.repeats,
            "model": args.model,
        },
        "caveats": "answerability는 proxy(정답성 보장 아님), tool-selection은 page_id→source 휴리스틱. 정답성(faithfulness/correctness)은 scripts/eval_generation.py(RAGAS, [eval] 설치)로 별도 측정.",
        "summaries": summaries,
        "aggregates": aggregates,
        "rows": all_rows,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsaved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
