"""GitHub 소스 fetcher — repo 문서(README·docs/)·이슈·PR을 MarkdownPage로 변환.

멀티소스 적재(#GitHub)의 fetch 어댑터. 인덱싱 downstream(chunker → embed → Qdrant/OpenSearch
``index_children``)은 Confluence와 **공유**한다(소스 무관). GitHub 콘텐츠는 이미 Markdown이라
HTML 클린 단계 없이 ``MarkdownPage``를 바로 생성한다.

출처 구분은 ``page_id`` 접두사(``gh:``)로 식별 가능. 정식 ``source`` 컬럼은 인덱싱 원장
일반화(confluence_document → source_document) PR에서 추가한다(#171 머지 후).
"""

from __future__ import annotations

import base64
import contextlib
import logging
from collections.abc import AsyncIterator

import httpx

from app.config import Settings, get_settings
from app.rag.chunker import MarkdownPage

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
_API_VERSION = "2022-11-28"


class GithubClient:
    """OnRamp org repo에서 문서(README·docs/)와 이슈/PR을 읽어 MarkdownPage로 변환.

    - ``fetch_repo_docs``: docs 전체 + README (온보딩·프로젝트 기술문서)
    - ``fetch_issues``: 이슈/PR + 코멘트 (장애대응·의사결정)
    인증: ``github_token``(repo scope). private repo도 토큰 권한 내에서 수집.
    """

    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings or get_settings()
        self.org = self.settings.github_org
        self._client = client  # 주입 시 테스트/재사용 (없으면 메서드별 생성)

    def _headers(self) -> dict[str, str]:
        token = self.settings.github_token.get_secret_value()
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": _API_VERSION}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _new_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=GITHUB_API, headers=self._headers(), timeout=30.0)

    @contextlib.asynccontextmanager
    async def _session(self) -> AsyncIterator[httpx.AsyncClient]:
        # 주입 클라는 닫지 않고 재사용(여러 메서드 호출 지원), 메서드별 생성 클라만 닫는다.
        if self._client is not None:
            yield self._client
        else:
            client = self._new_client()
            try:
                yield client
            finally:
                await client.aclose()

    # ── repo 문서 (README + docs/) ───────────────────────────────────────
    async def fetch_repo_docs(
        self, repo: str, *, include_readme: bool = True, docs_dirs: tuple[str, ...] | None = None
    ) -> list[MarkdownPage]:
        """repo의 ``README.md`` + ``docs/**.md``(또는 지정 디렉터리)를 MarkdownPage로 반환.

        '나머지 repo는 README, docs 레포는 전체' 정책: docs_dirs로 범위 조절.
        """
        if not repo:
            raise ValueError("repo must not be empty")
        dirs = docs_dirs if docs_dirs is not None else tuple(self.settings.github_docs_dirs)

        async with self._session() as client:
            branch = await self._default_branch(client, repo)
            paths = await self._list_markdown_paths(client, repo, branch)
            wanted = [p for p in paths if (include_readme and p.upper() == "README.MD") or _under_dirs(p, dirs)]
            pages: list[MarkdownPage] = []
            for path in wanted:
                content = await self._get_file_text(client, repo, path, branch)
                if content is None:
                    continue
                last_modified = await self._last_commit_date(client, repo, path, branch)
                pages.append(
                    MarkdownPage(
                        page_id=f"gh:{repo}:{path}",
                        page_title=path,
                        markdown=content,
                        source_url=f"https://github.com/{self.org}/{repo}/blob/{branch}/{path}",
                        space_key=repo,
                        last_modified=last_modified,
                    )
                )
        logger.info("GitHub repo docs fetched: %s (%d files)", repo, len(pages))
        return pages

    # ── 이슈 / PR (+코멘트) ──────────────────────────────────────────────
    async def fetch_issues(self, repo: str, *, state: str = "all", include_pr: bool = True) -> list[MarkdownPage]:
        """repo의 이슈/PR(+코멘트)을 MarkdownPage로 반환. PR은 issues 엔드포인트에 포함된다."""
        if not repo:
            raise ValueError("repo must not be empty")

        pages: list[MarkdownPage] = []
        async with self._session() as client:
            page = 1
            while True:
                resp = await client.get(
                    f"/repos/{self.org}/{repo}/issues",
                    params={"state": state, "per_page": 100, "page": page},
                )
                resp.raise_for_status()
                items = resp.json()
                if not items:
                    break
                for it in items:
                    is_pr = "pull_request" in it
                    if is_pr and not include_pr:
                        continue
                    comments = await self._issue_comments(client, repo, it["number"]) if it.get("comments") else ""
                    pages.append(_issue_to_page(repo, it, comments))
                page += 1
        logger.info("GitHub issues/PRs fetched: %s (%d items)", repo, len(pages))
        return pages

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────
    async def _default_branch(self, client: httpx.AsyncClient, repo: str) -> str:
        resp = await client.get(f"/repos/{self.org}/{repo}")
        resp.raise_for_status()
        return str(resp.json().get("default_branch", "main"))

    async def _list_markdown_paths(self, client: httpx.AsyncClient, repo: str, branch: str) -> list[str]:
        resp = await client.get(f"/repos/{self.org}/{repo}/git/trees/{branch}", params={"recursive": "1"})
        resp.raise_for_status()
        data = resp.json()
        if data.get("truncated"):
            logger.warning("GitHub tree truncated for %s — 일부 파일 누락 가능(대용량 repo)", repo)
        return [
            t["path"] for t in data.get("tree", []) if t.get("type") == "blob" and t["path"].lower().endswith(".md")
        ]

    async def _get_file_text(self, client: httpx.AsyncClient, repo: str, path: str, branch: str) -> str | None:
        resp = await client.get(f"/repos/{self.org}/{repo}/contents/{path}", params={"ref": branch})
        if resp.status_code != 200:
            logger.warning("GitHub file fetch 실패 %s:%s (%d)", repo, path, resp.status_code)
            return None
        data = resp.json()
        if data.get("encoding") != "base64" or "content" not in data:
            return None
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")

    async def _last_commit_date(self, client: httpx.AsyncClient, repo: str, path: str, branch: str) -> str:
        """파일 최신 커밋 시각(recency 부스트용). 실패 시 빈 문자열(best-effort)."""
        try:
            resp = await client.get(
                f"/repos/{self.org}/{repo}/commits", params={"path": path, "sha": branch, "per_page": 1}
            )
            resp.raise_for_status()
            commits = resp.json()
            if commits:
                return str(commits[0]["commit"]["committer"]["date"])
        except (httpx.HTTPError, KeyError, IndexError):
            logger.debug("last commit date 조회 실패 %s:%s", repo, path)
        return ""

    async def _issue_comments(self, client: httpx.AsyncClient, repo: str, number: int) -> str:
        resp = await client.get(f"/repos/{self.org}/{repo}/issues/{number}/comments", params={"per_page": 100})
        if resp.status_code != 200:
            return ""
        return "\n\n".join(f"**{c.get('user', {}).get('login', '?')}**: {c.get('body') or ''}" for c in resp.json())


def _under_dirs(path: str, dirs: tuple[str, ...]) -> bool:
    return any(path == d or path.startswith(f"{d}/") for d in dirs)


def _issue_to_page(repo: str, issue: dict, comments: str) -> MarkdownPage:
    kind = "PR" if "pull_request" in issue else "Issue"
    labels = ", ".join(label["name"] for label in issue.get("labels", []) if isinstance(label, dict))
    body = issue.get("body") or ""
    header = f"# [{kind} #{issue['number']}] {issue.get('title', '')}"
    meta = f"라벨: {labels}" if labels else ""
    markdown = "\n\n".join(part for part in (header, meta, body, comments) if part)
    return MarkdownPage(
        page_id=f"gh:{repo}#{issue['number']}",
        page_title=issue.get("title", ""),
        markdown=markdown,
        source_url=issue.get("html_url", ""),
        space_key=repo,
        last_modified=issue.get("updated_at", ""),
    )
