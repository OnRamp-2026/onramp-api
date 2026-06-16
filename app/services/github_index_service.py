"""GitHub 소스(README·docs·이슈/PR) → Qdrant·OpenSearch·PostgreSQL 색인.

Confluence 파이프라인(IngestService → IndexService)을 그대로 재사용한다:
  GithubClient.fetch_* → MarkdownPage
  → IngestService.prepare_github_pages (mask→profile→chunk→metadata 분류)
  → IndexService.index_prepared(source="github")  (Qdrant + OpenSearch 청크 + 원장 원문/registry)

원장(source_document)에는 source='github'로 적재되어 confluence와 page_id가 겹쳐도 분리된다.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from app.config import Settings, get_settings
from app.rag.chunker import MarkdownPage
from app.rag.sources.github import GithubClient
from app.services.index_service import IndexResult, IndexService
from app.services.ingest_service import IngestService

logger = logging.getLogger(__name__)


class GithubIndexService:
    """OnRamp org repo의 문서·이슈/PR을 RAG 인덱스에 적재."""

    def __init__(
        self,
        github: GithubClient | None = None,
        ingest: IngestService | None = None,
        index: IndexService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.github = github or GithubClient(settings=self.settings)
        self.ingest = ingest or IngestService(settings=self.settings)
        # IndexService가 ingest를 공유하도록 주입(분류기/마스커 재사용)
        self.index = index or IndexService(ingest_service=self.ingest, settings=self.settings)

    async def index_repos(
        self,
        repos: Sequence[str],
        *,
        include_docs: bool = True,
        include_issues: bool = True,
        include_pr: bool = True,
    ) -> IndexResult:
        """repos 목록의 문서/이슈를 모아 한 번의 index_run으로 적재."""
        pages: list[MarkdownPage] = []
        docs_dirs = tuple(self.settings.github_docs_dirs)
        for repo in repos:
            if include_docs:
                fetched = await self.github.fetch_repo_docs(repo, docs_dirs=docs_dirs or None)
                logger.info("github docs fetched: %s (%d)", repo, len(fetched))
                pages.extend(fetched)
            if include_issues:
                fetched = await self.github.fetch_issues(repo, include_pr=include_pr)
                logger.info("github issues fetched: %s (%d)", repo, len(fetched))
                pages.extend(fetched)

        if not pages:
            logger.warning("github: 적재할 페이지 없음 (repos=%s)", list(repos))
            return IndexResult(pages_indexed=0, chunks_indexed=0)

        prepared = self.ingest.prepare_github_pages(pages)
        return await self.index.index_prepared(prepared, source="github")
