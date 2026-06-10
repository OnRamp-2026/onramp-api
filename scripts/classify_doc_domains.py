"""문서 도메인 분류 dry-run CLI (Step 2, #49).

전체/제한 Confluence 페이지를 LLM으로 분류해 검수용 JSONL을 만든다. 색인/Qdrant에는 연결하지 않는다.
입력의 마스킹은 IngestService(masked_all_pages)가 처리하며, 이 스크립트는 마스킹하지 않는다.

전제: Confluence 접근 + OPENAI_API_KEY (분류용).
실행: PYTHONPATH=. python scripts/classify_doc_domains.py --limit 30 --output data/eval/doc_domain_dryrun.jsonl
"""

from __future__ import annotations

import argparse
import asyncio

from app.rag.doc_domain_classifier import DocumentDomainClassifier
from app.rag.doc_domain_dryrun import DryRunPage, load_existing, run_dry_run, write_jsonl
from app.services.ingest_service import IngestService

DEFAULT_OUTPUT = "data/eval/doc_domain_dryrun.jsonl"


async def main_async(limit: int, output: str, force: bool) -> None:
    masked_pages = await IngestService().masked_all_pages(limit=limit)
    pages = [
        DryRunPage(page_id=p.page_id, version=p.version, title=p.title, masked_markdown=p.markdown)
        for p in masked_pages
    ]
    existing = {} if force else load_existing(output)
    records, stats = await run_dry_run(pages, DocumentDomainClassifier(), existing=existing, force=force)
    write_jsonl(output, records)
    print(stats.as_line(), flush=True)
    print(f"→ {output} ({len(records)} records). 다음: 사람 검수(review_status pending→approved/edited)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="문서 도메인 분류 dry-run (#49 Step 2)")
    parser.add_argument("--limit", type=int, default=30, help="분류할 페이지 수 상한")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="검수용 JSONL 출력 경로")
    parser.add_argument("--force", action="store_true", help="기존 결과 재사용 없이 전부 재분류")
    args = parser.parse_args()
    asyncio.run(main_async(args.limit, args.output, args.force))


if __name__ == "__main__":
    main()
