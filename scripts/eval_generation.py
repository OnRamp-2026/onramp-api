"""생성 평가 하니스 CLI — 골든셋을 실 그래프로 흘려 답변 생성 후 RAGAS로 채점.

실측은 라이브 Qdrant + OpenAI(임베딩·생성·judge)를 사용한다(비용 발생).
LLM-judge는 비결정적이라 **회귀 게이트가 아니며**(nightly·수동), 추세 기록용 리포트만 남긴다.

답변 품질(RAGAS·answerability) 외에 운영 비용(token/latency)도 함께 잰다(#212). child-only↔
parent-expanded ablation은 `--parent-context`로 한 쌍의 재현 가능한 명령으로 돌린다.

사용:
    python scripts/eval_generation.py                          # answerable 전체 채점 점수표
    python scripts/eval_generation.py --limit 10               # 소규모 먼저(비용 절감)
    python scripts/eval_generation.py --write-report           # data/eval/gen_report.json 기록
    # #212 ablation (GPU 리랭커 strict): child-only vs parent-expanded
    python scripts/eval_generation.py --parent-context off --require-reranker --write-report --report off.json
    python scripts/eval_generation.py --parent-context on  --require-reranker --write-report --report on.json
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
from app.eval.generation_adapter import generate_for_eval  # noqa: E402
from app.eval.ragas_judge import ragas_available, resolve_judge_model, score_generation  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_REPORT = ROOT_DIR / "data" / "eval" / "gen_report.json"
RETRY_DELAY_SECONDS = 5  # 일시 오류(네트워크 등) 재시도 전 대기
MAX_CONSECUTIVE_FAILURES = 5  # 연속 실패 한도 — 일시 오류가 아닌 구조적 결함이면 조기 중단해 드러낸다


def _round(value: float, ndigits: int = 4) -> float:
    return round(value, ndigits)


def _percentile(values: list[float], pct: float) -> float:
    """정렬된 표본의 nearest-rank 백분위(p95 등). 빈 표본은 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * len(ordered) + 0.5) - 1))
    return ordered[idx]


def _cost_summary(results: list) -> dict:
    """운영 비용 축(#212) — 질의당 token/latency 평균 + latency p95. results는 생성 성공분."""
    n = len(results)
    if n == 0:
        return {"n": 0}
    latencies = [r.latency_s for r in results]
    return {
        "n": n,
        "avg_prompt_tokens": _round(sum(r.prompt_tokens for r in results) / n, 1),
        "avg_completion_tokens": _round(sum(r.completion_tokens for r in results) / n, 1),
        "avg_total_tokens": _round(sum(r.total_tokens for r in results) / n, 1),
        "avg_llm_calls": _round(sum(r.llm_calls for r in results) / n, 2),
        "latency_p50_s": _round(_percentile(latencies, 50), 3),
        "latency_p95_s": _round(_percentile(latencies, 95), 3),
        "total_tokens": sum(r.total_tokens for r in results),
    }


def _answerability_dist(results: list) -> dict:
    """답변 가능률 분포(#212 1차 결정 지표) — status별 카운트 + answerable 비율."""
    n = len(results)
    counts: dict[str, int] = {}
    for r in results:
        key = r.answerability_status or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return {
        "n": n,
        "counts": counts,
        "answerable_rate": _round(counts.get("answerable", 0) / n, 4) if n else 0.0,
        "not_enough_evidence_rate": _round(counts.get("not_enough_evidence", 0) / n, 4) if n else 0.0,
    }


def _rerank_summary(results: list) -> dict:
    """리랭커 발화 검증(#212 §2-5) — 폴백 건수와 발화 비율. 1.0이 아니면 GPU 측정이 오염됐다."""
    n = len(results)
    n_fallback = sum(1 for r in results if r.rerank_fallback)
    return {
        "n": n,
        "n_fallback": n_fallback,
        "rerank_fired_ratio": _round((n - n_fallback) / n, 4) if n else 0.0,
    }


async def run(args) -> int:
    if not ragas_available():
        logger.error('ragas 미설치 — 생성 평가를 건너뜁니다. 설치: uv pip install -e ".[eval]"')
        return 1

    settings = get_settings()
    # parent-expanded ablation(#212): 모드를 캐시된 settings 싱글톤에 박아 answer 노드까지 일관 적용.
    if args.parent_context is not None:
        settings.parent_context_enabled = args.parent_context == "on"
    mode = "parent-expanded" if settings.parent_context_enabled else "child-only"
    logger.info("컨텍스트 모드: %s (parent_context_enabled=%s)", mode, settings.parent_context_enabled)
    logger.info("리랭커 backend: %s (strict=%s)", settings.reranker_backend, args.require_reranker)

    golden = load_golden_set(args.queries, args.qrels)
    answerable = [g for g in golden if g.is_answerable]
    if args.limit is not None:  # truthy 체크면 --limit 0이 '무제한'처럼 동작 → is not None
        answerable = answerable[: max(0, args.limit)]  # 음수도 0건으로 안전 처리
    logger.info("생성 평가 대상 %d개 (answerable, limit=%s)", len(answerable), args.limit)

    def _gt(g) -> str | None:
        """공백뿐인 GT는 무효 — 채점(has_reference) 기준과 동일하게 처리."""
        return v if (v := (g.ground_truth_answer or "").strip()) else None

    n_with_gt = sum(1 for g in answerable if _gt(g))
    if args.with_reference and n_with_gt == 0:
        logger.warning("--with-reference 지정됐으나 GT 답변(ground_truth_answer)이 0건 — reference 지표는 건너뜀")

    results = []
    n_generation_failed = 0
    consecutive_failed = 0
    for i, g in enumerate(answerable, start=1):
        logger.info("[%d/%d] 생성: %.50s", i, len(answerable), g.query)
        reference = _gt(g) if args.with_reference else None
        # 장시간 실행(~1.5h) 중 일시 네트워크 오류 1건이 전체를 전멸시키지 않도록 문항 단위 방어 (#84).
        # 예외 타입 열거(재시도 가능만 선별) 대신 광역 catch + 연속 실패 서킷브레이커:
        # 생성 경로의 일시 오류는 LLMError 외에 임베더의 openai.APIConnectionError 등 다양해
        # 열거 누락 시 전손 모드가 재발하고, 구조적 결함은 연속 실패 한도가 조기에 드러낸다.
        for attempt in (1, 2):
            try:
                results.append(await generate_for_eval(g.query, domain=g.domain, model=args.model, reference=reference))
                consecutive_failed = 0
                break
            except Exception:
                if attempt == 1:
                    logger.warning("생성 실패 (qid=%s) — %ds 후 1회 재시도", g.qid, RETRY_DELAY_SECONDS, exc_info=True)
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                else:
                    n_generation_failed += 1
                    consecutive_failed += 1
                    logger.warning("생성 재시도 실패 (qid=%s) — 건너뜀", g.qid)
                    if consecutive_failed >= MAX_CONSECUTIVE_FAILURES:
                        raise RuntimeError(
                            f"연속 {MAX_CONSECUTIVE_FAILURES}개 문항 생성 실패 — 일시 오류가 아닌 "
                            "구조적 결함 가능성이 높아 중단합니다 (로그 traceback 확인)"
                        ) from None
    if n_generation_failed:
        logger.warning("생성 실패로 제외된 문항: %d건 (평가 분모에서 빠짐)", n_generation_failed)

    rerank = _rerank_summary(results)
    # 리랭커 strict 게이트(#212 §2-5): GPU 리랭커가 조용히 vector로 폴백하면 A/B가 오염된다.
    # 비싼 RAGAS 채점에 들어가기 전에 fail-loud — 한 건이라도 폴백되면 리포트를 내지 않는다.
    if args.require_reranker and rerank["n_fallback"] > 0:
        logger.error(
            "리랭커 strict 모드 위반 — %d/%d 문항이 vector로 폴백(rerank_fired_ratio=%.4f). "
            "GPU 리랭커 가동/URL을 확인하고 재실행하세요. 리포트 미발행.",
            rerank["n_fallback"],
            rerank["n"],
            rerank["rerank_fired_ratio"],
        )
        return 1

    cost = _cost_summary(results)
    answerability = _answerability_dist(results)

    scores = await score_generation(results, with_reference=args.with_reference)
    summary = scores.as_dict()

    print(f"\n=== 생성 평가 ({mode}) ===")
    print("[답변 가능률] (1차 결정 지표)")
    print(f"  answerable 비율        : {answerability['answerable_rate']}  ({answerability['counts']})")
    print(f"  not_enough_evidence 비율: {answerability['not_enough_evidence_rate']}")
    print("[RAGAS reference-free]")
    print(f"  Faithfulness     : {summary['faithfulness']}")
    print(f"  Answer Relevancy : {summary['answer_relevancy']}")
    print(f"  평가 샘플 n       : {summary['n_evaluated']}  (보류/무근거 제외: {summary['n_skipped']})")
    if args.with_reference:
        print("[reference 기반 (GT 답변 필요)]")
        print(f"  Factual Correctness : {summary['factual_correctness']}")
        print(f"  Semantic Similarity : {summary['semantic_similarity']}")
        print(f"  GT 보유 채점 n       : {summary['n_reference_evaluated']} / GT 보유 골든 {n_with_gt}")
    print("[운영 비용] (게이트 지표)")
    print(f"  질의당 prompt/completion/total tokens : {cost['avg_prompt_tokens']} / "
          f"{cost['avg_completion_tokens']} / {cost['avg_total_tokens']}")
    print(f"  latency p50 / p95 (s) : {cost['latency_p50_s']} / {cost['latency_p95_s']}")
    print(f"[리랭커] backend={settings.reranker_backend}  rerank_fired_ratio={rerank['rerank_fired_ratio']}"
          f"  (폴백 {rerank['n_fallback']}/{rerank['n']})")
    if rerank["n_fallback"] > 0:
        print("  ⚠️ 리랭커 폴백 발생 — GPU 측정이 일부 오염됨(strict 모드면 fail). 결과 신뢰 시 주의.")

    if args.write_report:
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "note": "LLM-judge(비결정) — 추세 기록용, 회귀 게이트 아님",
            "config": {
                "context_mode": mode,  # child-only | parent-expanded (#212 ablation arm)
                "parent_context_enabled": settings.parent_context_enabled,
                "reranker_backend": settings.reranker_backend,
                "require_reranker": args.require_reranker,
                "judge_model": resolve_judge_model(settings),  # 실제 채점에 쓰인 모델과 일치
                "embedding_model": settings.embedding_model,
                "n_golden_answerable": len([g for g in golden if g.is_answerable]),
                "limit": args.limit,
                "with_reference": args.with_reference,
                "n_generation_failed": n_generation_failed,
            },
            "generation": summary,
            "answerability": answerability,
            "cost": cost,
            "reranker": rerank,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("리포트 기록: %s", args.report)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="생성 평가 하니스 (RAGAS LLM-judge, 비차단).")
    parser.add_argument("--queries", type=Path, default=ROOT_DIR / "data" / "eval" / "queries.jsonl")
    parser.add_argument("--qrels", type=Path, default=ROOT_DIR / "data" / "eval" / "qrels.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="평가 문항 수 제한(비용 절감)")
    parser.add_argument("--model", default="", help="답변 생성 모델(빈값=config 기본)")
    parser.add_argument(
        "--with-reference",
        action="store_true",
        help="GT 답변 기반 지표(FactualCorrectness/SemanticSimilarity) 추가 채점 (#67, GT 있는 문항만)",
    )
    parser.add_argument(
        "--parent-context",
        choices=["on", "off"],
        default=None,
        help="컨텍스트 모드 강제(#212 ablation). on=parent-expanded, off=child-only. 미지정=config 기본값",
    )
    parser.add_argument(
        "--require-reranker",
        action="store_true",
        help="리랭커 strict 모드(#212): 한 문항이라도 vector 폴백되면 fail(리포트 미발행). GPU 리랭커 측정 보장",
    )
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
