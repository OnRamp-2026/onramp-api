"""GithubClient — GitHub API를 MockTransport로 목킹해 MarkdownPage 변환 검증."""

import base64

import httpx
import pytest

from app.config import Settings
from app.rag.chunker import MarkdownPage
from app.rag.sources.github import GITHUB_API, GithubClient


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def _docs_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/repos/OnRamp-2026/onramp":
        return httpx.Response(200, json={"default_branch": "main"})
    if path == "/repos/OnRamp-2026/onramp/git/trees/main":
        return httpx.Response(
            200,
            json={
                "truncated": False,
                "tree": [
                    {"path": "README.md", "type": "blob"},
                    {"path": "docs/guide.md", "type": "blob"},
                    {"path": "docs/sub/deep.md", "type": "blob"},
                    {"path": "src/app.py", "type": "blob"},  # 비-md → 제외
                    {"path": "notes.md", "type": "blob"},  # docs 밖·README 아님 → 제외
                ],
            },
        )
    if path.startswith("/repos/OnRamp-2026/onramp/contents/"):
        fname = path.rsplit("/", 1)[-1]
        return httpx.Response(200, json={"encoding": "base64", "content": _b64(f"# {fname}\n본문")})
    if path == "/repos/OnRamp-2026/onramp/commits":
        return httpx.Response(200, json=[{"commit": {"committer": {"date": "2026-06-10T00:00:00Z"}}}])
    return httpx.Response(404, json={})


def _client_with(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=GITHUB_API, transport=httpx.MockTransport(handler))


def _settings() -> Settings:
    return Settings(github_org="OnRamp-2026", github_docs_dirs=["docs"])


async def test_fetch_repo_docs_keeps_readme_and_docs_only():
    gh = GithubClient(settings=_settings(), client=_client_with(_docs_handler))
    pages = await gh.fetch_repo_docs("onramp")

    paths = {p.page_id for p in pages}
    assert paths == {
        "gh:onramp:README.md",
        "gh:onramp:docs/guide.md",
        "gh:onramp:docs/sub/deep.md",
    }  # src/app.py(비md)·notes.md(docs밖) 제외
    page = next(p for p in pages if p.page_id == "gh:onramp:docs/guide.md")
    assert isinstance(page, MarkdownPage)
    assert page.markdown.startswith("# guide.md")
    assert page.source_url == "https://github.com/OnRamp-2026/onramp/blob/main/docs/guide.md"
    assert page.space_key == "onramp"
    assert page.last_modified == "2026-06-10T00:00:00Z"


async def test_fetch_repo_docs_readme_can_be_excluded():
    gh = GithubClient(settings=_settings(), client=_client_with(_docs_handler))
    pages = await gh.fetch_repo_docs("onramp", include_readme=False)
    assert all(not p.page_id.endswith("README.md") for p in pages)


def _issues_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/repos/OnRamp-2026/onramp/issues":
        if request.url.params.get("page") == "1":
            return httpx.Response(
                200,
                json=[
                    {
                        "number": 10,
                        "title": "장애: OOM",
                        "body": "리랭커 OOM 발생",
                        "html_url": "https://github.com/OnRamp-2026/onramp/issues/10",
                        "updated_at": "2026-06-15T00:00:00Z",
                        "labels": [{"name": "incident"}],
                        "comments": 1,
                    },
                    {
                        "number": 11,
                        "title": "PR: 수정",
                        "body": "fix",
                        "html_url": "https://github.com/OnRamp-2026/onramp/pull/11",
                        "updated_at": "2026-06-16T00:00:00Z",
                        "labels": [],
                        "comments": 0,
                        "pull_request": {"url": "..."},
                    },
                ],
            )
        return httpx.Response(200, json=[])  # page 2 → 종료
    if path == "/repos/OnRamp-2026/onramp/issues/10/comments":
        return httpx.Response(200, json=[{"user": {"login": "minji"}, "body": "노드 메모리 부족"}])
    return httpx.Response(404, json={})


async def test_fetch_issues_includes_pr_and_comments():
    gh = GithubClient(settings=_settings(), client=_client_with(_issues_handler))
    pages = await gh.fetch_issues("onramp")

    assert {p.page_id for p in pages} == {"gh:onramp#10", "gh:onramp#11"}
    issue = next(p for p in pages if p.page_id == "gh:onramp#10")
    assert "[Issue #10]" in issue.markdown and "incident" in issue.markdown
    assert "**minji**: 노드 메모리 부족" in issue.markdown  # 코멘트 병합
    pr = next(p for p in pages if p.page_id == "gh:onramp#11")
    assert "[PR #11]" in pr.markdown


async def test_fetch_issues_can_skip_pr():
    gh = GithubClient(settings=_settings(), client=_client_with(_issues_handler))
    pages = await gh.fetch_issues("onramp", include_pr=False)
    assert {p.page_id for p in pages} == {"gh:onramp#10"}  # PR(#11) 제외


async def test_empty_repo_raises():
    gh = GithubClient(settings=_settings(), client=_client_with(_docs_handler))
    with pytest.raises(ValueError):
        await gh.fetch_repo_docs("")
