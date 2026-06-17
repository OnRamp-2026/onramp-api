"""GitHub 적재 서비스 단위 테스트 (네트워크/LLM 없이 fake로)."""

from __future__ import annotations

from app.rag.chunker import MarkdownPage
from app.services.github_index_service import GithubIndexService
from app.services.index_service import IndexResult
from app.services.ingest_service import IngestService


def _page(page_id: str, title: str = "T", md: str = "# h\n본문") -> MarkdownPage:
    return MarkdownPage(page_id=page_id, page_title=title, markdown=md, source_url="https://x", space_key="onramp-api")


def test_github_to_cleaned_preserves_markdown_as_raw() -> None:
    ingest = IngestService()
    mp = _page("gh:onramp-api:README.md", title="README.md", md="# 온보딩\n내용")
    cleaned = ingest._github_to_cleaned(mp)

    assert cleaned.page_id == "gh:onramp-api:README.md"
    assert cleaned.title == "README.md"
    assert cleaned.space_key == "onramp-api"
    assert cleaned.markdown == "# 온보딩\n내용"
    assert cleaned.html == "# 온보딩\n내용"  # GitHub 원문 = Markdown (raw_html에 보존)
    assert cleaned.version is None


class _FakeGithub:
    def __init__(self) -> None:
        self.docs_calls: list[str] = []
        self.issue_calls: list[str] = []

    async def fetch_repo_docs(self, repo: str, *, docs_dirs=None):  # noqa: ANN001
        self.docs_calls.append(repo)
        return [_page(f"gh:{repo}:README.md")]

    async def fetch_issues(self, repo: str, *, include_pr: bool = True):
        self.issue_calls.append(repo)
        return [_page(f"gh:{repo}#1", title="issue")]


class _FakeIndex:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    async def index_prepared(self, pages, *, source: str = "confluence") -> IndexResult:  # noqa: ANN001
        self.calls.append((len(pages), source))
        return IndexResult(pages_indexed=len(pages), chunks_indexed=len(pages) * 2)


class _FakeIngest:
    async def prepare_github_pages(self, pages):  # noqa: ANN001
        # 변환/청킹은 다른 테스트가 검증 — 여기선 길이 보존만
        return list(pages)


async def test_index_repos_aggregates_and_tags_github_source() -> None:
    github = _FakeGithub()
    index = _FakeIndex()
    service = GithubIndexService(github=github, ingest=_FakeIngest(), index=index)  # type: ignore[arg-type]

    result = await service.index_repos(["onramp-api", "onramp-web"])

    # 2 repos × (1 doc + 1 issue) = 4 pages, source는 반드시 github
    assert github.docs_calls == ["onramp-api", "onramp-web"]
    assert github.issue_calls == ["onramp-api", "onramp-web"]
    assert index.calls == [(4, "github")]
    assert result.pages_indexed == 4


async def test_index_repos_empty_skips_index() -> None:
    class _Empty(_FakeGithub):
        async def fetch_repo_docs(self, repo, *, docs_dirs=None):  # noqa: ANN001
            return []

        async def fetch_issues(self, repo, *, include_pr: bool = True):
            return []

    index = _FakeIndex()
    service = GithubIndexService(github=_Empty(), ingest=_FakeIngest(), index=index)  # type: ignore[arg-type]
    result = await service.index_repos(["onramp-api"])

    assert index.calls == []  # 페이지 없으면 index 호출 안 함
    assert result.pages_indexed == 0
