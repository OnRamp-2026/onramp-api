"""Deterministic vs Single Agentic RAG full-graph A/B evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.agents.graph import compiled_graph  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.eval.dataset import load_golden_set  # noqa: E402
from app.eval.metrics import aggregate  # noqa: E402
from app.services.llm_selector import usage_accumulator  # noqa: E402


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * ratio) - 1)
    return ordered[index]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    quality = aggregate([(list(row["ranked_chunk_ids"]), set(row["relevant_chunk_ids"])) for row in rows]).as_dict()
    latencies = [float(row["latency_ms"]) for row in rows]
    traces = [trace for row in rows for trace in row.get("tool_trace", [])]
    retries = [int(row.get("retry_count", 0)) for row in rows]
    tokens = [int(row.get("tokens", 0)) for row in rows]
    return {
        "runs": len(rows),
        "quality": quality,
        "latency_ms": {
            "avg": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p50": round(_percentile(latencies, 0.50), 2),
            "p95": round(_percentile(latencies, 0.95), 2),
        },
        "avg_tool_calls": round(len(traces) / len(rows), 4) if rows else 0.0,
        "retry_ratio": round(sum(value > 0 for value in retries) / len(rows), 4) if rows else 0.0,
        "fallback_ratio": (
            round(sum(bool(trace.get("fallback")) for trace in traces) / len(traces), 4) if traces else 0.0
        ),
        "avg_tokens": round(sum(tokens) / len(rows), 2) if rows else 0.0,
    }


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    golden = [item for item in load_golden_set(args.queries, args.qrels) if item.is_answerable]
    if args.limit:
        golden = golden[: args.limit]
    rows: list[dict[str, Any]] = []
    for strategy in ("deterministic", "single_agentic"):
        for repeat in range(args.repeats):
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
                with usage_accumulator() as usage:
                    result = await compiled_graph.ainvoke(state)
                rows.append(
                    {
                        "strategy": strategy,
                        "repeat": repeat + 1,
                        "qid": item.qid,
                        "latency_ms": round((perf_counter() - started) * 1000, 2),
                        "ranked_chunk_ids": [
                            document.chunk_id for document in result.get("documents", []) if document.chunk_id
                        ],
                        "relevant_chunk_ids": list(item.relevant_chunk_ids),
                        "tool_trace": [
                            vars(trace) if hasattr(trace, "__dict__") else trace
                            for trace in result.get("tool_trace", [])
                        ],
                        "retry_count": result.get("retry_count", 0),
                        "tokens": usage["total"],
                    }
                )
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {"queries": len(golden), "repeats": args.repeats, "model": args.model},
        "deterministic": summarize([row for row in rows if row["strategy"] == "deterministic"]),
        "single_agentic": summarize([row for row in rows if row["strategy"] == "single_agentic"]),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("deterministic", "single_agentic")}, ensure_ascii=False, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=ROOT_DIR / "data/eval/queries.jsonl")
    parser.add_argument("--qrels", type=Path, default=ROOT_DIR / "data/eval/qrels.jsonl")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default="")
    parser.add_argument("--tenant-id", default="")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "data/eval/results/single_agentic_ab.json")
    raise SystemExit(asyncio.run(_run(parser.parse_args())))


if __name__ == "__main__":
    main()
