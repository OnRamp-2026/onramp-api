"""룰 vs 문서 단위 LLM 도메인 분류 — 공통 ground truth 대비 정확도 맞대결.

같은 문서에 대해 세 라벨을 구한다:
  - 룰 라벨   = OpenSearch 청크 다수결 domain (현재 색인된 값)
  - LLM 라벨  = DocumentDomainClassifier (gpt-4o-mini, 구현본)
  - 정답(best) = 더 강한 심사자(gpt-4o)가 **라벨을 모른 채** 독립적으로 고른 best 도메인

룰·LLM 각각이 정답과 얼마나 맞는지(정확도)와 승부(LLM만 정답 / 룰만 정답 / …)를 출력한다.
적재 데이터를 바꾸지 않는다(읽기 전용). 마스킹된 본문만 LLM에 보낸다.

예) python scripts/benchmark_rule_vs_llm.py --limit 100
    python scripts/benchmark_rule_vs_llm.py --source confluence --limit 80 --truth-model gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import httpx
from sqlalchemy import func, select

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings  # noqa: E402
from app.db.models import SourceDocument  # noqa: E402
from app.db.postgres import session_scope  # noqa: E402
from app.rag.llm_classifier import DOMAIN_DEFINITIONS, DocumentDomainClassifier  # noqa: E402
from app.rag.masker import MarkdownMasker  # noqa: E402
from app.services.llm_selector import call_llm  # noqa: E402

logger = logging.getLogger(__name__)
DOMAINS = list(DOMAIN_DEFINITIONS)

# 정답 라벨러 — 분류기 라벨을 보여주지 않고 독립적으로 best 도메인을 고르게 한다(편향 차단).
_TRUTH_SYSTEM = (
    "너는 사내 문서를 아래 5개 도메인 중 가장 적합한 하나로 분류하는 전문가다.\n\n"
    + "\n".join(f"- {k}: {v}" for k, v in DOMAIN_DEFINITIONS.items())
    + "\n\n판정 기준:\n"
    + "- 회의록·기획 형식 문서는 형식 우선(meeting_note/planning).\n"
    + "- incident는 실제 장애 사후분석(postmortem·outage)에만.\n"
    + "- api_reference는 REST/HTTP API 엔드포인트·요청/응답 스펙일 때만. 서버/미들웨어 모듈·지시어\n"
    + "  설정 문서(Apache/nginx mod_* 등)는 manual.\n"
    + '반드시 JSON만: {"best": "<5개 중 1>"}'
)

# 편향 제거(neutral) 정답 라벨러 — tie-break 규칙 없이 도메인 정의만 준다. 분류기와 공유하는
# 채점 기준을 제거해, "정책 주입 효과"를 뺀 더 공정한 룰/LLM 격차를 본다.
_TRUTH_SYSTEM_NEUTRAL = (
    "너는 사내 문서를 아래 5개 도메인 중 가장 적합한 하나로 분류하는 전문가다.\n\n"
    + "\n".join(f"- {k}: {v}" for k, v in DOMAIN_DEFINITIONS.items())
    + '\n\n반드시 JSON만: {"best": "<5개 중 1>"}'
)


async def _rule_domain_by_page(base: str) -> dict[str, str]:
    body = {"size": 10000, "_source": ["page_id", "domain"], "query": {"match_all": {}}}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{base}/onramp-chunks/_search", json=body)
        r.raise_for_status()
        hits = r.json()["hits"]["hits"]
    votes: dict[str, Counter] = defaultdict(Counter)
    for h in hits:
        s = h["_source"]
        votes[s.get("page_id")][s.get("domain")] += 1
    return {pid: c.most_common(1)[0][0] for pid, c in votes.items() if c}


async def _sample_documents(source: str | None, limit: int) -> list[SourceDocument]:
    async with session_scope() as db:
        stmt = select(SourceDocument).order_by(func.random()).limit(limit * 3)
        if source:
            stmt = select(SourceDocument).where(SourceDocument.source == source).order_by(func.random()).limit(limit * 3)
        rows = (await db.execute(stmt)).scalars().all()
    return [r for r in rows if (r.cleaned_markdown or "").strip()][:limit]


async def _truth(title: str, masked: str, model: str, *, neutral: bool = False) -> str | None:
    system = _TRUTH_SYSTEM_NEUTRAL if neutral else _TRUTH_SYSTEM
    user = f"제목: {title}\n\n본문:\n{masked[:1800]}"
    try:
        raw = await call_llm(system, user, model=model, temperature=0.0, max_tokens=60, json_mode=True)
        best = json.loads(raw).get("best")
    except Exception as exc:  # noqa: BLE001
        logger.warning("정답 라벨 실패: %s", exc)
        return None
    return best if best in DOMAINS else None


async def run(source: str | None, limit: int, truth_model: str, concurrency: int, neutral: bool) -> None:
    s = get_settings()
    base = f"{s.opensearch_scheme}://{s.opensearch_host}:{s.opensearch_port}"
    rule_by_page = await _rule_domain_by_page(base)
    docs = [d for d in await _sample_documents(source, limit) if d.page_id in rule_by_page]
    mode = "neutral(편향제거)" if neutral else "policy(tie-break 포함)"
    logger.info("벤치마크 문서 %d개 (source=%s, truth=%s, 정답기준=%s)", len(docs), source or "전체", truth_model, mode)

    masker = MarkdownMasker()
    classifier = DocumentDomainClassifier()
    sem = asyncio.Semaphore(concurrency)

    async def _one(doc: SourceDocument):
        async with sem:
            masked = masker.mask(doc.cleaned_markdown or "")
            llm = await classifier.classify(doc.title, masked)
            truth = await _truth(doc.title, masked, truth_model, neutral=neutral)
            if llm is None or truth is None:
                return None
            return rule_by_page[doc.page_id], llm.domain, truth, doc

    rows = [r for r in await asyncio.gather(*(_one(d) for d in docs)) if r]
    n = len(rows)
    if not n:
        logger.warning("유효 결과 없음")
        return

    rule_ok = sum(1 for rule, _, truth, _ in rows if rule == truth)
    llm_ok = sum(1 for _, llm, truth, _ in rows if llm == truth)
    both = sum(1 for rule, llm, truth, _ in rows if rule == truth and llm == truth)
    llm_only = sum(1 for rule, llm, truth, _ in rows if llm == truth and rule != truth)
    rule_only = sum(1 for rule, llm, truth, _ in rows if rule == truth and llm != truth)
    neither = sum(1 for rule, llm, truth, _ in rows if rule != truth and llm != truth)

    by_truth: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # truth → [rule_ok, llm_ok, total]
    for rule, llm, truth, _ in rows:
        st = by_truth[truth]
        st[2] += 1
        st[0] += int(rule == truth)
        st[1] += int(llm == truth)

    print("\n" + "=" * 64)
    print(f"룰 vs LLM 도메인 분류 성능 — 정답=gpt-4o, {n}건")
    print("=" * 64)
    print(f"\n  룰  정확도: {rule_ok:3d}/{n} = {rule_ok / n * 100:5.1f}%")
    print(f"  LLM 정확도: {llm_ok:3d}/{n} = {llm_ok / n * 100:5.1f}%")
    print(f"  → 개선폭: {(llm_ok - rule_ok) / n * 100:+.1f}p")

    print("\n[승부]")
    print(f"  LLM만 정답: {llm_only:3d}    룰만 정답: {rule_only:3d}")
    print(f"  둘 다 정답: {both:3d}    둘 다 오답: {neither:3d}")

    print("\n[정답 도메인별 정확도 — 룰 / LLM]")
    print(f"    {'domain':14s} {'룰':>8s} {'LLM':>8s}  N")
    for d in DOMAINS:
        ro, lo, tot = by_truth[d]
        if tot:
            print(f"    {d:14s} {ro / tot * 100:7.0f}% {lo / tot * 100:7.0f}%  {tot}")

    print("\n[룰 오답 → LLM 교정 샘플 8건]")
    shown = 0
    for rule, llm, truth, doc in rows:
        if llm == truth and rule != truth and shown < 8:
            print(f"    정답={truth:13s} 룰={rule:13s} ({doc.source}) {doc.title[:40]!r}")
            shown += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="룰 vs 문서 단위 LLM 분류 정확도 맞대결(정답=gpt-4o, 읽기 전용).")
    parser.add_argument("--source", default=None, help="confluence | github (미지정 시 전체)")
    parser.add_argument("--limit", type=int, default=100, help="벤치마크 문서 수")
    parser.add_argument("--truth-model", default="gpt-4o", help="정답 라벨러 모델(분류기보다 강한 모델)")
    parser.add_argument(
        "--neutral-truth",
        action="store_true",
        help="정답 라벨러에서 tie-break 규칙 제거(도메인 정의만) — 분류기와 공유 기준 제거한 공정 비교",
    )
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    asyncio.run(run(args.source, args.limit, args.truth_model, args.concurrency, args.neutral_truth))


if __name__ == "__main__":
    main()
