"""검색 평가 하니스 CLI — 골든셋으로 dense/rerank 검색 품질을 결정론적으로 측정.

실측은 라이브 Qdrant + OpenAI 임베딩(쿼리당 1회)을 사용한다(비용 발생).
출력: 모드별 Hit Rate@k·MRR@k·Recall@k·nDCG@k 점수표 + dense→rerank 델타 + answerability 정확도.

사용:
    python scripts/eval_retrieval.py                       # dense,rerank 점수표
    python scripts/eval_retrieval.py --write-baseline      # data/eval/baseline.json 고정
    python scripts/eval_retrieval.py --gate                # baseline 대비 회귀 시 exit 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings  # noqa: E402
from app.eval.dataset import load_golden_set  # noqa: E402
from app.eval.metrics import MetricSummary, aggregate, answerability_accuracy  # noqa: E402
from app.eval.retrieval_adapter import Mode, predicted_answerable, retrieve_for_eval  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_BASELINE = ROOT_DIR / "data" / "eval" / "baseline.json"
GATED_MODE = "rerank"  # 회귀 게이트 기준 모드 (운영 경로)


async def _eval_mode(golden, mode: Mode, *, top_k, top_n, ans_floor, ans_min_docs):
    """한 모드로 전체 골든셋을 검색해 (검색 요약, answerability 요약)을 반환."""
    per_query: list[tuple[list[str], set[str]]] = []
    preds: list[bool] = []
    labels: list[bool] = []
    for g in golden:
        result = await retrieve_for_eval(g.query, mode=mode, domain=g.domain, top_k=top_k, top_n=top_n)
        per_query.append((result.chunk_ids, set(g.relevant_chunk_ids)))
        preds.append(predicted_answerable(result, floor=ans_floor, min_docs=ans_min_docs))
        labels.append(g.is_answerable)
    return aggregate(per_query), answerability_accuracy(preds, labels)


def _print_table(summaries: dict[str, MetricSummary]) -> None:
    header = f"{'metric':<14}" + "".join(f"{m:>12}" for m in summaries)
    print("\n" + header)
    print("-" * len(header))
    sample = next(iter(summaries.values()))
    for label, key in [
        (f"Hit Rate@{sample.k_hit}", f"hit_rate@{sample.k_hit}"),
        (f"MRR@{sample.k_mrr}", f"mrr@{sample.k_mrr}"),
        (f"Recall@{sample.k_recall}", f"recall@{sample.k_recall}"),
        (f"nDCG@{sample.k_ndcg}", f"ndcg@{sample.k_ndcg}"),
    ]:
        row = f"{label:<14}" + "".join(f"{s.as_dict()[key]:>12.4f}" for s in summaries.values())
        print(row)
    print(f"\nn(평가 대상 질문) = {sample.n}")

    if "dense" in summaries and "rerank" in summaries:
        d, r = summaries["dense"].as_dict(), summaries["rerank"].as_dict()
        print("\ndense → rerank 델타:")
        for key in d:
            print(f"  {key:<12} {d[key]:+.4f} → {r[key]:+.4f}  (Δ {r[key] - d[key]:+.4f})")


def _build_report(summaries, ans, *, top_k, top_n) -> dict:
    settings = get_settings()
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "n": next(iter(summaries.values())).n,
        "config": {
            "top_k": top_k,
            "top_n": top_n,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
        },
        **{mode: summary.as_dict() for mode, summary in summaries.items()},
        "answerability": ans.as_dict(),
    }


def _check_gate(report: dict, baseline_path: Path, tolerance: float) -> int:
    if not baseline_path.exists():
        logger.error("baseline 없음: %s (먼저 --write-baseline)", baseline_path)
        return 1
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    base = baseline.get(GATED_MODE, {})
    curr = report.get(GATED_MODE, {})
    regressions = []
    for key, base_val in base.items():
        curr_val = curr.get(key, 0.0)
        if curr_val < base_val - tolerance:
            regressions.append((key, base_val, curr_val))
    if regressions:
        logger.error("회귀 감지 (%s, tolerance=%.3f):", GATED_MODE, tolerance)
        for key, b, c in regressions:
            logger.error("  %s: baseline %.4f → 현재 %.4f", key, b, c)
        return 1
    logger.info("게이트 통과 — %s 결정론 지표 회귀 없음", GATED_MODE)
    return 0


async def run(args) -> int:
    golden = load_golden_set(args.queries, args.qrels)
    logger.info("골든셋 %d개 로드 (모드: %s)", len(golden), ", ".join(args.modes))

    summaries: dict[str, MetricSummary] = {}
    ans = None
    for mode in args.modes:
        summary, ans_mode = await _eval_mode(
            golden,
            mode,
            top_k=args.top_k,
            top_n=args.top_n,
            ans_floor=args.ans_floor,
            ans_min_docs=args.ans_min_docs,
        )
        summaries[mode] = summary
        if mode == GATED_MODE or ans is None:
            ans = ans_mode

    _print_table(summaries)
    print(f"\nAnswerability: {ans.as_dict()}  (tp={ans.tp} fp={ans.fp} tn={ans.tn} fn={ans.fn})")

    report = _build_report(summaries, ans, top_k=args.top_k, top_n=args.top_n)

    if args.write_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("baseline 기록: %s", args.baseline)

    if args.gate:
        return _check_gate(report, args.baseline, args.tolerance)
    return 0


def _parse_modes(value: str) -> list[Mode]:
    modes = [m.strip() for m in value.split(",") if m.strip()]
    valid = {"dense", "rerank"}
    bad = [m for m in modes if m not in valid]
    if bad:
        raise argparse.ArgumentTypeError(f"지원하지 않는 mode: {bad} (가능: {sorted(valid)})")
    return modes  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description="검색 평가 하니스 (결정론 지표).")
    parser.add_argument("--queries", type=Path, default=ROOT_DIR / "data" / "eval" / "queries.jsonl")
    parser.add_argument("--qrels", type=Path, default=ROOT_DIR / "data" / "eval" / "qrels.jsonl")
    parser.add_argument("--modes", type=_parse_modes, default="dense,rerank", help="쉼표 구분 (dense,rerank)")
    parser.add_argument("--top-k", type=int, default=None, help="Qdrant 후보 풀 (기본: config)")
    parser.add_argument("--top-n", type=int, default=None, help="최종 top-N (기본: config)")
    parser.add_argument("--ans-floor", type=float, default=0.0, help="answerable 예측 점수 임계값 τ (보정 전 0)")
    parser.add_argument("--ans-min-docs", type=int, default=1)
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--gate", action="store_true", help="baseline 대비 회귀 시 exit 1")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if isinstance(args.modes, str):  # default 문자열 처리
        args.modes = _parse_modes(args.modes)

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
