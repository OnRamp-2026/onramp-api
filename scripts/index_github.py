"""GitHub repo 문서·이슈/PR을 Qdrant·OpenSearch·PostgreSQL에 색인.

예) 전체:        python scripts/index_github.py --repos onramp-api onramp-web
    문서만:      python scripts/index_github.py --repos onramp-api --no-issues
    이슈만:      python scripts/index_github.py --repos onramp-api --no-docs

토큰: GITHUB_TOKEN(repo scope) — private repo 포함. 조직: GITHUB_ORG(기본 OnRamp-2026).
OpenSearch 청크 적재는 BM25_SEARCH_ENABLED=true일 때만(기본 off → Qdrant·Postgres만).
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

from app.services.github_index_service import GithubIndexService  # noqa: E402

logger = logging.getLogger(__name__)


async def run(repos: list[str], *, docs: bool, issues: bool, include_pr: bool, force: bool = False) -> None:
    service = GithubIndexService()
    result = await service.index_repos(
        repos, include_docs=docs, include_issues=issues, include_pr=include_pr, force=force
    )
    logger.info(
        "GitHub 색인 완료: %s pages, %s chunks (repos=%s)",
        result.pages_indexed,
        result.chunks_indexed,
        repos,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GitHub repo 문서·이슈/PR을 RAG 인덱스에 적재.")
    parser.add_argument("--repos", nargs="+", required=True, help="repo 이름 목록(org 제외, 예: onramp-api)")
    parser.add_argument("--no-docs", action="store_false", dest="docs", help="README·docs 제외")
    parser.add_argument("--no-issues", action="store_false", dest="issues", help="이슈/PR 제외")
    parser.add_argument("--no-pr", action="store_false", dest="include_pr", help="PR 제외(이슈만)")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="content-hash dedup 무시하고 전체 재색인(도메인 분류만 바꿔 다시 분류·임베딩). 전체 wipe 불필요.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    asyncio.run(run(args.repos, docs=args.docs, issues=args.issues, include_pr=args.include_pr, force=args.reindex))


if __name__ == "__main__":
    main()
