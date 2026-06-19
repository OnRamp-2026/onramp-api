"""deterministic/agentic Retriever A/B 평가.

동일 골든셋을 두 전략으로 반복 실행해 검색 품질, 지연시간, Agentic 도구 선택과
fallback 비율을 JSON·Markdown으로 저장한다. Router와 Answer는 제외하고
`retrieve_with_diagnostics`를 호출해 Retriever 검색·rerank 경로만 비교한다.

실행:
    python scripts/eval_agentic_retriever.py
    python scripts/eval_agentic_retriever.py --repeats 3 --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.agents.retriever.node import retrieve_with_diagnostics  # noqa: E402
from app.agents.state import AgentState, Domain  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.eval.dataset import GoldenQuery, load_golden_set  # noqa: E402
from app.eval.metrics import aggregate  # noqa: E402

logger = logging.getLogger(__name__)

Strategy = Literal["deterministic", "agentic"]
DEFAULT_JSON = ROOT_DIR / "data" / "eval" / "results" / "agentic_retriever_ab.json"
DEFAULT_MARKDOWN = ROOT_DIR / "data" / "eval" / "results" / "agentic_retriever_ab.md"


def _round(value: float, digits: int = 4) -> float:
    return round(value, digits)


def _percentile(values: list[float], quantile: float) -> float:
    """선형 보간 percentile. 표본 1개도 안전하게 처리한다."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _aggregate_strategy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_query = [(list(row["ranked_chunk_ids"]), set(row["relevant_chunk_ids"])) for row in rows]
    quality = aggregate(per_query).as_dict()
    latencies = [float(row["latency_ms"]) for row in rows]
    tool_sets: list[set[str]] = []
    tool_calls: list[int] = []
    fallback_reasons: Counter[str] = Counter()
    duplicate_calls = 0
    rrf_count = 0
    rerank_fallback_count = 0

    for row in rows:
        diagnostics = row["diagnostics"]
        tools = {str(call.get("tool", "")) for call in diagnostics.get("calls", []) if call.get("tool")}
        tool_sets.append(tools)
        tool_calls.append(int(diagnostics.get("tool_call_count", 0)))
        duplicate_calls += int(diagnostics.get("duplicate_calls", 0))
        rrf_count += int(bool(diagnostics.get("rrf_applied")))
        rerank_fallback_count += int(bool(diagnostics.get("rerank_fallback")))
        if diagnostics.get("fallback"):
            fallback_reasons[str(diagnostics["fallback"])] += 1

    n = len(rows)
    agentic = {
        "avg_tool_calls": _round(sum(tool_calls) / n) if n else 0.0,
        "dense_selection_ratio": _round(sum("search_dense" in tools for tools in tool_sets) / n) if n else 0.0,
        "bm25_selection_ratio": _round(sum("search_bm25" in tools for tools in tool_sets) / n) if n else 0.0,
        "multi_tool_ratio": _round(sum(len(tools) >= 2 for tools in tool_sets) / n) if n else 0.0,
        "rrf_ratio": _round(rrf_count / n) if n else 0.0,
        "fallback_ratio": _round(sum(fallback_reasons.values()) / n) if n else 0.0,
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "duplicate_call_count": duplicate_calls,
        "rerank_fallback_ratio": _round(rerank_fallback_count / n) if n else 0.0,
    }
    return {
        "runs": n,
        "quality": quality,
        "latency_ms": {
            "avg": _round(sum(latencies) / n, 2) if n else 0.0,
            "p50": _round(_percentile(latencies, 0.50), 2),
            "p95": _round(_percentile(latencies, 0.95), 2),
        },
        "agentic": agentic,
    }


def _build_report(
    rows: list[dict[str, Any]],
    *,
    repeats: int,
    model: str,
    tenant_id: str,
) -> dict[str, Any]:
    deterministic = _aggregate_strategy([row for row in rows if row["strategy"] == "deterministic"])
    agentic = _aggregate_strategy([row for row in rows if row["strategy"] == "agentic"])
    det_quality = deterministic["quality"]
    agent_quality = agentic["quality"]
    delta = {key: _round(agent_quality[key] - det_quality[key]) for key in det_quality}
    delta["avg_latency_ms"] = _round(
        agentic["latency_ms"]["avg"] - deterministic["latency_ms"]["avg"],
        2,
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "repeats": repeats,
            "model": model,
            "tenant_id": tenant_id,
            "query_count": len({row["qid"] for row in rows}),
        },
        "deterministic": deterministic,
        "agentic": agentic,
        "delta": delta,
        "runs": rows,
    }


def _signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def _markdown_report(report: dict[str, Any]) -> str:
    det = report["deterministic"]
    agent = report["agentic"]
    delta = report["delta"]
    quality_rows = [
        ("Hit Rate@5", "hit_rate@5"),
        ("Recall@5", "recall@5"),
        ("MRR@10", "mrr@10"),
        ("nDCG@10", "ndcg@10"),
    ]
    lines = [
        "# Agentic Retriever A/B 평가",
        "",
        f"- 생성 시각: `{report['generated_at']}`",
        f"- 질문 수: `{report['config']['query_count']}`",
        f"- 전략별 반복 수: `{report['config']['repeats']}`",
        f"- 모델: `{report['config']['model'] or '(config default)'}`",
        f"- tenant: `{report['config']['tenant_id']}`",
        "",
        "## 요약",
        "",
        "| 지표 | deterministic | agentic | delta |",
        "|---|---:|---:|---:|",
    ]
    for label, key in quality_rows:
        lines.append(f"| {label} | {det['quality'][key]:.4f} | {agent['quality'][key]:.4f} | {_signed(delta[key])} |")
    lines.extend(
        [
            (
                "| 평균 지연(ms) | "
                f"{det['latency_ms']['avg']:.2f} | {agent['latency_ms']['avg']:.2f} | "
                f"{_signed(delta['avg_latency_ms'], 2)} |"
            ),
            f"| p50 지연(ms) | {det['latency_ms']['p50']:.2f} | {agent['latency_ms']['p50']:.2f} | - |",
            f"| p95 지연(ms) | {det['latency_ms']['p95']:.2f} | {agent['latency_ms']['p95']:.2f} | - |",
            "",
            "## Agentic 실행 진단",
            "",
            f"- 평균 tool 호출 수: `{agent['agentic']['avg_tool_calls']:.4f}`",
            f"- dense 선택률: `{agent['agentic']['dense_selection_ratio']:.4f}`",
            f"- BM25 선택률: `{agent['agentic']['bm25_selection_ratio']:.4f}`",
            f"- 복수 도구 선택률: `{agent['agentic']['multi_tool_ratio']:.4f}`",
            f"- RRF 적용률: `{agent['agentic']['rrf_ratio']:.4f}`",
            f"- deterministic fallback 비율: `{agent['agentic']['fallback_ratio']:.4f}`",
            f"- rerank fallback 비율: `{agent['agentic']['rerank_fallback_ratio']:.4f}`",
            f"- fallback 원인: `{json.dumps(agent['agentic']['fallback_reasons'], ensure_ascii=False)}`",
            "",
            "상세 질문별 실행 결과는 같은 이름의 JSON 파일에서 확인한다.",
            "",
        ]
    )
    return "\n".join(lines)


def _state_for(golden: GoldenQuery, *, model: str, tenant_id: str) -> AgentState:
    domains = [Domain(golden.domain)] if golden.domain else []
    return {
        "query": golden.query,
        "refined_query": golden.query,
        "domains": domains,
        "target_versions": [],
        "model": model,
        "tenant_id": tenant_id,
    }


async def _run_once(
    golden: GoldenQuery,
    *,
    strategy: Strategy,
    repeat: int,
    model: str,
    tenant_id: str,
    settings: Settings,
) -> dict[str, Any]:
    strategy_settings = settings.model_copy(update={"retriever_strategy": strategy})
    started = perf_counter()
    output, diagnostics = await retrieve_with_diagnostics(
        _state_for(golden, model=model, tenant_id=tenant_id),
        settings=strategy_settings,
    )
    latency_ms = (perf_counter() - started) * 1000
    documents = output["documents"]
    return {
        "strategy": strategy,
        "qid": golden.qid,
        "query": golden.query,
        "repeat": repeat,
        "latency_ms": _round(latency_ms, 2),
        "ranked_chunk_ids": [doc.chunk_id for doc in documents if doc.chunk_id],
        "relevant_chunk_ids": list(golden.relevant_chunk_ids),
        "diagnostics": diagnostics.as_dict(),
    }


async def run(args: argparse.Namespace) -> int:
    if args.repeats <= 0:
        raise ValueError("--repeats는 1 이상이어야 합니다")
    golden = load_golden_set(args.queries, args.qrels)
    if args.answerable_only:
        golden = [g for g in golden if g.is_answerable and g.relevant_chunk_ids]
    if args.limit is not None:
        golden = golden[: args.limit]
    if not golden:
        raise ValueError("평가할 질문이 없습니다")

    settings = get_settings()
    tenant_id = args.tenant_id or settings.auth_default_tenant
    rows: list[dict[str, Any]] = []
    total = len(golden) * args.repeats * 2
    current = 0
    for strategy in ("deterministic", "agentic"):
        for repeat in range(1, args.repeats + 1):
            for item in golden:
                current += 1
                logger.info("[%d/%d] %s repeat=%d qid=%s", current, total, strategy, repeat, item.qid)
                rows.append(
                    await _run_once(
                        item,
                        strategy=strategy,
                        repeat=repeat,
                        model=args.model,
                        tenant_id=tenant_id,
                        settings=settings,
                    )
                )

    report = _build_report(rows, repeats=args.repeats, model=args.model, tenant_id=tenant_id)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(_markdown_report(report), encoding="utf-8")
    print(_markdown_report(report))
    logger.info("JSON 저장: %s", args.output_json)
    logger.info("Markdown 저장: %s", args.output_markdown)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="deterministic/agentic Retriever A/B 평가")
    parser.add_argument("--queries", type=Path, default=ROOT_DIR / "data/eval/queries.jsonl")
    parser.add_argument("--qrels", type=Path, default=ROOT_DIR / "data/eval/qrels.jsonl")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="앞에서 N개 질문만 빠르게 실행")
    parser.add_argument("--model", default="", help="tool calling 모델. 비우면 config 기본 모델")
    parser.add_argument("--tenant-id", default="", help="평가 tenant. 비우면 AUTH_DEFAULT_TENANT")
    parser.add_argument(
        "--answerable-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="정답 chunk가 있는 answerable 질문만 실행",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
