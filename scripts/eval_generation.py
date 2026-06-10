"""생성 평가 하니스 CLI — 골든셋을 실 그래프로 흘려 답변 생성 후 RAGAS로 채점.

실측은 라이브 Qdrant + OpenAI(임베딩·생성·judge)를 사용한다(비용 발생).
LLM-judge는 비결정적이라 **회귀 게이트가 아니며**(nightly·수동), 추세 기록용 리포트만 남긴다.

사용:
    python scripts/eval_generation.py                      # answerable 전체 채점 점수표
    python scripts/eval_generation.py --limit 10           # 소규모 먼저(비용 절감)
    python scripts/eval_generation.py --write-report       # data/eval/gen_report.json 기록
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


async def run(args) -> int:
    if not ragas_available():
        logger.error('ragas 미설치 — 생성 평가를 건너뜁니다. 설치: uv pip install -e ".[eval]"')
        return 1

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
    for i, g in enumerate(answerable, start=1):
        logger.info("[%d/%d] 생성: %.50s", i, len(answerable), g.query)
        reference = _gt(g) if args.with_reference else None
        # 장시간 실행(~1.5h) 중 일시 네트워크 오류 1건이 전체를 전멸시키지 않도록 문항 단위 방어 (#84)
        for attempt in (1, 2):
            try:
                results.append(await generate_for_eval(g.query, domain=g.domain, model=args.model, reference=reference))
                break
            except Exception:
                if attempt == 1:
                    logger.warning("생성 실패 (qid=%s) — %ds 후 1회 재시도", g.qid, RETRY_DELAY_SECONDS, exc_info=True)
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                else:
                    n_generation_failed += 1
                    logger.warning("생성 재시도 실패 (qid=%s) — 건너뜀", g.qid)
    if n_generation_failed:
        logger.warning("생성 실패로 제외된 문항: %d건 (평가 분모에서 빠짐)", n_generation_failed)

    scores = await score_generation(results, with_reference=args.with_reference)
    summary = scores.as_dict()

    print("\n=== 생성 평가 (RAGAS) ===")
    print("[reference-free]")
    print(f"  Faithfulness     : {summary['faithfulness']}")
    print(f"  Answer Relevancy : {summary['answer_relevancy']}")
    print(f"  평가 샘플 n       : {summary['n_evaluated']}  (보류/무근거 제외: {summary['n_skipped']})")
    if args.with_reference:
        print("[reference 기반 (GT 답변 필요)]")
        print(f"  Factual Correctness : {summary['factual_correctness']}")
        print(f"  Semantic Similarity : {summary['semantic_similarity']}")
        print(f"  GT 보유 채점 n       : {summary['n_reference_evaluated']} / GT 보유 골든 {n_with_gt}")

    if args.write_report:
        settings = get_settings()
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "note": "LLM-judge(비결정) — 추세 기록용, 회귀 게이트 아님",
            "config": {
                "judge_model": resolve_judge_model(settings),  # 실제 채점에 쓰인 모델과 일치
                "embedding_model": settings.embedding_model,
                "n_golden_answerable": len([g for g in golden if g.is_answerable]),
                "limit": args.limit,
                "with_reference": args.with_reference,
                "n_generation_failed": n_generation_failed,
            },
            "generation": summary,
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
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
