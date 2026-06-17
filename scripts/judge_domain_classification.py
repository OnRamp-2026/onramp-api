"""LLM 분류 품질을 더 강한 LLM 심사자로 평가 (LLM-as-judge).

구현된 분류기(``DocumentDomainClassifier``, gpt-4o-mini)가 매긴 도메인이 타당한지를,
**다른(더 강한) 모델**(기본 gpt-4o)이 독립적으로 평가한다. 자기검증 편향을 줄이려 심사자는:
  1. 문서를 보고 스스로 best 도메인을 고르고(strict 일치 측정)
  2. 분류기 라벨이 정의상 방어 가능한지(acceptable) 판정한다(lenient 측정)

적재 데이터를 바꾸지 않는다(읽기 전용). 마스킹된 본문만 LLM에 보낸다.

예) python scripts/judge_domain_classification.py --limit 80
    python scripts/judge_domain_classification.py --source confluence --limit 60 --judge-model gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sqlalchemy import func, select

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.db.models import SourceDocument  # noqa: E402
from app.db.postgres import session_scope  # noqa: E402
from app.rag.llm_classifier import DOMAIN_DEFINITIONS, DocumentDomainClassifier  # noqa: E402
from app.rag.masker import MarkdownMasker  # noqa: E402
from app.services.llm_selector import call_llm  # noqa: E402

logger = logging.getLogger(__name__)
DOMAINS = list(DOMAIN_DEFINITIONS)

_JUDGE_SYSTEM = (
    "너는 문서 도메인 분류 결과를 검증하는 엄격한 심사자다. 5개 도메인 정의:\n\n"
    + "\n".join(f"- {k}: {v}" for k, v in DOMAIN_DEFINITIONS.items())
    + "\n\n문서(제목·본문)와 '분류기가 매긴 라벨'이 주어진다. 다음을 판정하라:\n"
    + "1. best: 네가 보기에 가장 적합한 도메인 1개.\n"
    + "2. acceptable: 분류기 라벨이 정의상 방어 가능하면 true (best와 달라도 충분히 타당하면 true).\n"
    + "3. reason: 한 문장 근거.\n"
    + '반드시 JSON만: {"best": "<5개 중 1>", "acceptable": <true|false>, "reason": "<한 문장>"}'
)


async def _sample_documents(source: str | None, limit: int) -> list[SourceDocument]:
    async with session_scope() as db:
        stmt = select(SourceDocument).order_by(func.random()).limit(limit * 3)
        if source:
            stmt = (
                select(SourceDocument).where(SourceDocument.source == source).order_by(func.random()).limit(limit * 3)
            )
        rows = (await db.execute(stmt)).scalars().all()
    return [r for r in rows if (r.cleaned_markdown or "").strip()][:limit]


async def _judge(title: str, masked: str, assigned: str, model: str) -> dict | None:
    user = f"제목: {title}\n분류기 라벨: {assigned}\n\n본문:\n{masked[:1800]}"
    try:
        raw = await call_llm(_JUDGE_SYSTEM, user, model=model, temperature=0.0, max_tokens=200, json_mode=True)
        data = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("심사 실패: %s", exc)
        return None
    if not isinstance(data, dict) or data.get("best") not in DOMAINS:
        return None
    # acceptable을 bool로 강제 — 문자열 "false"가 truthy로 잡혀 lenient 지표를 오염시키지 않게.
    acceptable = data.get("acceptable")
    if isinstance(acceptable, str):
        acceptable = acceptable.strip().lower() == "true"
    data["acceptable"] = bool(acceptable)
    return data


async def run(source: str | None, limit: int, judge_model: str, concurrency: int) -> None:
    docs = await _sample_documents(source, limit)
    logger.info("심사 대상 문서 %d개 (source=%s, judge=%s) — 분류→심사 시작", len(docs), source or "전체", judge_model)

    masker = MarkdownMasker()
    classifier = DocumentDomainClassifier()
    sem = asyncio.Semaphore(concurrency)

    async def _one(doc: SourceDocument) -> tuple[str, dict, SourceDocument] | None:
        async with sem:
            masked = masker.mask(doc.cleaned_markdown or "")
            result = await classifier.classify(doc.title, masked)
            if result is None:
                return None
            verdict = await _judge(doc.title, masked, result.domain, judge_model)
            if verdict is None:
                return None
            return result.domain, verdict, doc

    rows = [r for r in await asyncio.gather(*(_one(d) for d in docs)) if r]
    n = len(rows)
    if not n:
        logger.warning("유효 심사 결과 없음")
        return

    strict = sum(1 for assigned, v, _ in rows if v["best"] == assigned)
    lenient = sum(1 for _, v, _ in rows if v.get("acceptable"))
    per_domain: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # assigned → [strict, acceptable, total]
    wrong_moves: Counter = Counter()
    for assigned, v, _ in rows:
        st = per_domain[assigned]
        st[2] += 1
        if v["best"] == assigned:
            st[0] += 1
        if v.get("acceptable"):
            st[1] += 1
        else:
            wrong_moves[(assigned, v["best"])] += 1

    print("\n" + "=" * 66)
    print(f"LLM 분류 품질 심사 (judge={judge_model}) — 유효 {n}건")
    print("=" * 66)
    print(f"\nstrict 일치(심사자 best == 분류기): {strict}/{n} = {strict / n * 100:.1f}%")
    print(f"lenient 타당(분류기 라벨 acceptable): {lenient}/{n} = {lenient / n * 100:.1f}%")

    print("\n[분류기 라벨별 품질]")
    for d in DOMAINS:
        st, ac, tot = per_domain[d]
        if tot:
            print(
                f"    {d:14s} strict {st:3d}/{tot:<3d} ({st / tot * 100:4.0f}%)  | acceptable {ac:3d}/{tot:<3d} ({ac / tot * 100:4.0f}%)"
            )

    print("\n[부적합 판정 — 분류기 라벨 → 심사자 best : 횟수]")
    for (assigned, best), cnt in wrong_moves.most_common(12):
        print(f"    {assigned:14s} → {best:14s} : {cnt}")

    print("\n[부적합 샘플 8건]")
    shown = 0
    for assigned, v, doc in rows:
        if not v.get("acceptable") and shown < 8:
            print(f"    [{assigned} → {v['best']}] ({doc.source}) {doc.title[:42]!r}")
            print(f"        근거: {v.get('reason', '')[:90]}")
            shown += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 도메인 분류를 더 강한 LLM 심사자로 평가(읽기 전용).")
    parser.add_argument("--source", default=None, help="confluence | github (미지정 시 전체)")
    parser.add_argument("--limit", type=int, default=80, help="심사할 문서 수")
    parser.add_argument("--judge-model", default="gpt-4o", help="심사자 모델(분류기보다 강한 모델 권장)")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    asyncio.run(run(args.source, args.limit, args.judge_model, args.concurrency))


if __name__ == "__main__":
    main()
