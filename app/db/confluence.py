"""Confluence Cloud API client."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast
from urllib.parse import parse_qsl, urlsplit
from zoneinfo import ZoneInfo

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfluencePage:
    """A Confluence page with storage HTML content."""

    page_id: str
    title: str
    space_key: str
    html: str
    last_modified: str
    version: int | None
    url: str


class ConfluenceClient:
    """Read and update pages from the configured Confluence space."""

    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client

    async def fetch_recent_pages(self, hours: int = 24, limit: int = 50) -> list[ConfluencePage]:
        """Return pages modified within the last N hours."""

        if hours <= 0:
            raise ValueError("hours must be greater than 0")
        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        since = datetime.now(ZoneInfo(self.settings.confluence_timezone)) - timedelta(hours=hours)
        cql = self._build_recent_pages_cql(since)
        logger.info("Searching Confluence with CQL: %s", cql)
        return await self._search_pages(cql=cql, limit=limit)

    async def fetch_all_pages(self, limit: int = 100) -> list[ConfluencePage]:
        """Return every page in the configured space (no lastmodified filter).

        Used for the initial full ingestion (`--all`), unlike fetch_recent_pages
        which only returns pages modified within the last N hours.
        """

        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        cql = self._build_pages_cql(since=None)
        logger.info("Searching Confluence with CQL: %s", cql)
        return await self._search_pages(cql=cql, limit=limit)

    async def fetch_candidate_pages(self, limit: int = 100) -> list[ConfluencePage]:
        """Return pages from the configured space for test mutation (same query as fetch_all_pages)."""

        return await self.fetch_all_pages(limit=limit)

    async def create_page(self, title: str, html: str, space_key: str | None = None) -> ConfluencePage:
        """Create a new Confluence page (storage HTML) and return it."""

        if not title or not title.strip():
            raise ValueError("Confluence page title must not be empty")

        payload = {
            "type": "page",
            "title": title,
            "space": {"key": space_key or self.settings.confluence_space_key},
            "body": {
                "storage": {
                    "value": html,
                    "representation": "storage",
                }
            },
        }
        result = await self._request_json("POST", "/content", json=payload)
        return self._to_page(result)

    async def update_page(self, page: ConfluencePage, updated_html: str, next_version: int) -> None:
        """Update a Confluence page storage body."""

        payload = {
            "id": page.page_id,
            "type": "page",
            "title": page.title,
            "space": {"key": page.space_key},
            "body": {
                "storage": {
                    "value": updated_html,
                    "representation": "storage",
                }
            },
            "version": {"number": next_version},
        }
        await self._request_json("PUT", f"/content/{page.page_id}", json=payload)

    async def _search_pages(self, cql: str, limit: int) -> list[ConfluencePage]:
        """CQL 검색 결과를 `_links.next` 커서로 페이지네이션한다.

        Confluence Cloud는 `/content/search`에서 `start` 오프셋을 **무시**하고 항상
        첫 페이지를 반환한다(실측: start=0~850 모두 동일 결과). 오프셋 방식으로는
        limit이 page_size를 넘는 순간 같은 페이지를 중복 수집하므로, 응답의
        `_links.next`(cursor 포함)를 따라가는 방식만 유효하다.
        """
        pages: list[ConfluencePage] = []
        seen_page_ids: set[str] = set()
        params: dict[str, Any] = {
            "cql": cql,
            "expand": "body.storage,version,space",
            "limit": min(limit, 50),
        }

        while len(pages) < limit:
            payload = await self._request_json("GET", "/content/search", params=params)
            results = payload.get("results", [])
            if not results:
                break

            for result in results:
                page = self._to_page(result)
                if page.page_id in seen_page_ids:  # 서버측 중복 반환 방어
                    continue
                seen_page_ids.add(page.page_id)
                pages.append(page)
                if len(pages) >= limit:
                    return pages

            next_link = (payload.get("_links") or {}).get("next")
            if not next_link:
                break
            # next 링크의 쿼리(cursor·cql·limit 등)를 그대로 다음 요청 파라미터로 사용
            params = dict(parse_qsl(urlsplit(next_link).query))

        return pages

    async def _request_json(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        client = self.client
        if client is not None:
            response = await client.request(method, f"{self.rest_api_base_url}{path}", params=params, json=json)
            response.raise_for_status()
            return self._json_object(response)

        async with httpx.AsyncClient(
            auth=(self.settings.confluence_user_email, self.settings.confluence_api_token),
            headers={"Accept": "application/json", "User-Agent": "OnRamp-RAG/0.1"},
            timeout=30,
        ) as scoped_client:
            response = await scoped_client.request(method, f"{self.rest_api_base_url}{path}", params=params, json=json)
            response.raise_for_status()
            return self._json_object(response)

    def _json_object(self, response: httpx.Response) -> dict[str, Any]:
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("Confluence API response must be a JSON object")
        return cast(dict[str, Any], payload)

    def _build_recent_pages_cql(self, since: datetime) -> str:
        return self._build_pages_cql(since=since)

    def _build_pages_cql(self, since: datetime | None) -> str:
        """Build the page-search CQL. With `since`, filter by lastmodified (recent);
        without it, return the whole space ordered by title (full ingestion)."""
        space_key = self._quote_cql_value(self.settings.confluence_space_key)
        if since is None:
            # id ASC 타이브레이커 — 동일 title 다수일 때 커서 페이지네이션 순서가 안정적이도록
            return f'type = page AND space = "{space_key}" ORDER BY title ASC, id ASC'
        since_text = since.strftime("%Y-%m-%d %H:%M")
        return f'type = page AND space = "{space_key}" AND lastmodified >= "{since_text}" ORDER BY lastmodified DESC'

    def _to_page(self, result: dict[str, Any]) -> ConfluencePage:
        content = result.get("content", result)
        body = content.get("body", {})
        storage = body.get("storage", {})
        version = content.get("version", {})
        space = content.get("space", {})
        return ConfluencePage(
            page_id=str(content.get("id", "")),
            title=content.get("title", ""),
            space_key=space.get("key", self.settings.confluence_space_key),
            html=storage.get("value", ""),
            last_modified=version.get("when", ""),
            version=version.get("number"),
            url=self._build_page_url(content, result),
        )

    def _build_page_url(self, content: dict[str, Any], result: dict[str, Any]) -> str:
        links = content.get("_links", {}) or result.get("_links", {})
        webui = links.get("webui") or result.get("url")
        if not isinstance(webui, str) or not webui:
            return ""
        if webui.startswith("http"):
            return webui
        base = self.settings.confluence_base_url.removesuffix("/wiki")
        return f"{base}/wiki{webui}"

    def _quote_cql_value(self, value: str) -> str:
        return value.replace('"', '\\"')

    @property
    def rest_api_base_url(self) -> str:
        base_url = self.settings.confluence_base_url.rstrip("/")
        if base_url.endswith("/wiki"):
            return f"{base_url}/rest/api"
        return f"{base_url}/wiki/rest/api"
