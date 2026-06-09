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
from app.eval.ragas_judge import ragas_available, score_generation  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_REPORT = ROOT_DIR / "data" / "eval" / "gen_report.json"


async def run(args) -> int:
    if not ragas_available():
        logger.error('ragas 미설치 — 생성 평가를 건너뜁니다. 설치: uv pip install -e ".[eval]"')
        return 1

    golden = load_golden_set(args.queries, args.qrels)
    answerable = [g for g in golden if g.is_answerable]
    if args.limit:
        answerable = answerable[: args.limit]
    logger.info("생성 평가 대상 %d개 (answerable, limit=%s)", len(answerable), args.limit)

    results = []
    for i, g in enumerate(answerable, start=1):
        logger.info("[%d/%d] 생성: %.50s", i, len(answerable), g.query)
        results.append(await generate_for_eval(g.query, domain=g.domain, model=args.model))

    scores = await score_generation(results)
    summary = scores.as_dict()

    print("\n=== 생성 평가 (RAGAS, reference-free) ===")
    print(f"  Faithfulness     : {summary['faithfulness']}")
    print(f"  Answer Relevancy : {summary['answer_relevancy']}")
    print(f"  평가 샘플 n       : {summary['n_evaluated']}  (보류/무근거 제외: {summary['n_skipped']})")

    if args.write_report:
        settings = get_settings()
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "note": "LLM-judge(비결정) — 추세 기록용, 회귀 게이트 아님",
            "config": {
                "judge_model": settings.default_model or "gpt-4o-mini",
                "embedding_model": settings.embedding_model,
                "n_golden_answerable": len([g for g in golden if g.is_answerable]),
                "limit": args.limit,
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
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
