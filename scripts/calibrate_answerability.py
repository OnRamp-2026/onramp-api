"""answerability 임계값 τ 보정 — 골든셋의 rerank top_score 분포로 최적 τ를 찾는다.

각 질문을 rerank 검색해 1위 점수를 모으고, is_answerable 라벨 대비 τ를 스윕해
precision/recall/F1/Youden's J 를 계산한다. 권장 τ는 Youden's J(=TPR-FPR) 최대값
(동률 시 precision 높은 쪽). 이 τ를 eval_retrieval.py 의 --ans-floor 기본값으로 쓰고,
#B Trust 재검색 트리거도 동일 신호를 재사용한다.

의존: 라이브 Qdrant + OpenAI 임베딩 + 리랭커. (비용 발생)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.eval.dataset import load_golden_set  # noqa: E402
from app.eval.metrics import AnswerabilitySummary, answerability_accuracy  # noqa: E402
from app.eval.retrieval_adapter import retrieve_for_eval  # noqa: E402

logger = logging.getLogger(__name__)

SweepRow = tuple[float, AnswerabilitySummary, float]  # (τ, 지표, Youden)


async def _collect(golden, top_k, top_n) -> list[tuple[float, bool, int]]:
    """(top_score, is_answerable, n_docs) 수집 (rerank 모드)."""
    rows: list[tuple[float, bool, int]] = []
    for g in golden:
        r = await retrieve_for_eval(g.query, mode="rerank", domain=g.domain, top_k=top_k, top_n=top_n)
        rows.append((r.top_score, g.is_answerable, r.n))
    return rows


def _sweep(rows: list[tuple[float, bool, int]], min_docs: int) -> tuple[list[SweepRow], tuple | None]:
    labels = [ans for _, ans, _ in rows]
    scores = sorted({s for s, _, _ in rows})
    # 후보 τ: 관측 점수들 사이 midpoint + 양 끝
    cands = [scores[0] - 1e-6] + [(scores[i] + scores[i + 1]) / 2 for i in range(len(scores) - 1)]
    best = None
    table = []
    for tau in cands:
        preds = [(s >= tau and n >= min_docs) for s, _, n in rows]
        m = answerability_accuracy(preds, labels)
        fpr = m.fp / (m.fp + m.tn) if (m.fp + m.tn) else 0.0
        youden = m.recall - fpr  # TPR - FPR
        table.append((tau, m, youden))
        key = (round(youden, 6), round(m.precision, 6))
        if best is None or key > best[0]:
            best = (key, tau, m, youden)
    return table, best


async def run(args) -> None:
    golden = load_golden_set(args.queries, args.qrels)
    ans = sum(1 for g in golden if g.is_answerable)
    logger.info("골든셋 %d (answerable %d / unanswerable %d)", len(golden), ans, len(golden) - ans)

    rows = await _collect(golden, args.top_k, args.top_n)
    pos = [s for s, a, _ in rows if a]
    neg = [s for s, a, _ in rows if not a]
    if not pos:
        logger.error("골든셋에 answerable 쿼리가 없습니다 — 보정 불가")
        return
    print("\n[top_score 분포]")
    print(f"  answerable   n={len(pos)}  min={min(pos):.3f}  mean={sum(pos) / len(pos):.3f}  max={max(pos):.3f}")
    if neg:
        print(f"  unanswerable n={len(neg)}  min={min(neg):.3f}  mean={sum(neg) / len(neg):.3f}  max={max(neg):.3f}")

    table, best = _sweep(rows, args.min_docs)
    print("\n[τ 스윕]  τ        acc    prec   recall  f1     Youden")
    for tau, m, y in table:
        mark = " ◀ best" if best and abs(tau - best[1]) < 1e-12 else ""
        print(f"  {tau:8.3f}  {m.accuracy:.3f}  {m.precision:.3f}  {m.recall:.3f}  {m.f1:.3f}  {y:+.3f}{mark}")

    _, tau, m, y = best
    print(
        f"\n권장 τ = {tau:.4f}  (Youden={y:+.3f}, acc={m.accuracy:.3f}, prec={m.precision:.3f}, recall={m.recall:.3f})"
    )
    print(f"  → eval_retrieval.py --ans-floor {tau:.4f}  (또는 ANSWERABILITY_FLOOR 기본값으로 반영)")


def main() -> None:
    parser = argparse.ArgumentParser(description="answerability τ 보정.")
    parser.add_argument("--queries", type=Path, default=ROOT_DIR / "data" / "eval" / "queries.jsonl")
    parser.add_argument("--qrels", type=Path, default=ROOT_DIR / "data" / "eval" / "qrels.jsonl")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--min-docs", type=int, default=1)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
