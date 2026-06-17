"""적재된 데이터의 룰 도메인 vs 문서 단위 LLM 분류 비교.

구현된 기능(문서 단위 ``DocumentDomainClassifier``, LLM_CLASSIFY_ENABLED)이 현재 색인된
룰 라벨과 얼마나 다른지 실측한다. 각 문서마다:

  - 룰 도메인 = OpenSearch 청크들의 다수결 ``domain`` (현재 색인된 값)
  - LLM 도메인 = DocumentDomainClassifier.classify(title, 마스킹된 cleaned_markdown)

일치율·혼동행렬·분포 변화·불일치 샘플을 출력한다. 적재 데이터를 바꾸지 않는다(읽기 전용).

예) python scripts/compare_domain_classification.py --limit 150
    python scripts/compare_domain_classification.py --source confluence --limit 100
"""

from __future__ import annotations

import argparse
import asyncio
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

logger = logging.getLogger(__name__)
DOMAINS = list(DOMAIN_DEFINITIONS)


async def _rule_domain_by_page(base: str) -> dict[str, str]:
    """OpenSearch 청크에서 page_id → 다수결 룰 도메인 맵을 만든다."""
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
            stmt = (
                select(SourceDocument).where(SourceDocument.source == source).order_by(func.random()).limit(limit * 3)
            )
        rows = (await db.execute(stmt)).scalars().all()
    return [r for r in rows if (r.cleaned_markdown or "").strip()][:limit]


async def run(source: str | None, limit: int, concurrency: int) -> None:
    s = get_settings()
    base = f"{s.opensearch_scheme}://{s.opensearch_host}:{s.opensearch_port}"

    rule_by_page = await _rule_domain_by_page(base)
    docs = await _sample_documents(source, limit)
    docs = [d for d in docs if d.page_id in rule_by_page]
    logger.info("비교 대상 문서 %d개 (source=%s) — 문서 단위 LLM 분류 시작", len(docs), source or "전체")

    masker = MarkdownMasker()
    classifier = DocumentDomainClassifier()
    sem = asyncio.Semaphore(concurrency)

    async def _one(doc: SourceDocument) -> tuple[str, str | None, SourceDocument]:
        async with sem:
            masked = masker.mask(doc.cleaned_markdown or "")
            result = await classifier.classify(doc.title, masked)
            return rule_by_page[doc.page_id], (result.domain if result else None), doc

    rows = await asyncio.gather(*(_one(d) for d in docs))
    pairs = [(rule, llm, d) for rule, llm, d in rows if llm]

    n = len(pairs)
    if not n:
        logger.warning("유효 비교 결과 없음")
        return
    agree = sum(1 for rule, llm, _ in pairs if rule == llm)
    confusion: Counter = Counter()
    rule_dist: Counter = Counter()
    llm_dist: Counter = Counter()
    for rule, llm, _ in pairs:
        rule_dist[rule] += 1
        llm_dist[llm] += 1
        if rule != llm:
            confusion[(rule, llm)] += 1

    print("\n" + "=" * 66)
    print(f"룰(청크 다수결) ↔ 문서 단위 LLM 분류 비교 — 유효 {n}건 (분류 실패 {len(docs) - n})")
    print("=" * 66)
    print(f"\n전체 일치율: {agree}/{n} = {agree / n * 100:.1f}%\n")

    print("[도메인 분포 변화 — 룰 → LLM]")
    for d in DOMAINS:
        rc, lc = rule_dist.get(d, 0), llm_dist.get(d, 0)
        arrow = "→" if rc == lc else ("↑" if lc > rc else "↓")
        print(f"    {d:14s} {rc:4d}  {arrow}  {lc:4d}")

    print("\n[주요 라벨 이동 (룰 → LLM): 횟수]")
    for (rule, llm), cnt in confusion.most_common(12):
        print(f"    {rule:14s} → {llm:14s} : {cnt}")

    print("\n[불일치 문서 샘플 10건]")
    shown = 0
    for rule, llm, d in pairs:
        if rule != llm and shown < 10:
            print(f"    [{rule} → {llm}] ({d.source}) {d.title[:48]!r}")
            shown += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="적재된 룰 도메인 vs 문서 단위 LLM 분류 비교(읽기 전용).")
    parser.add_argument("--source", default=None, help="confluence | github (미지정 시 전체)")
    parser.add_argument("--limit", type=int, default=120, help="비교할 문서 수")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    asyncio.run(run(args.source, args.limit, args.concurrency))


if __name__ == "__main__":
    main()
