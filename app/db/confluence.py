"""Confluence Cloud API client."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast
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

    async def fetch_candidate_pages(self, limit: int = 100) -> list[ConfluencePage]:
        """Return recent pages from the configured space for test mutation."""

        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        cql = (
            f'type = page AND space = "{self._quote_cql_value(self.settings.confluence_space_key)}" ORDER BY title ASC'
        )
        return await self._search_pages(cql=cql, limit=limit)

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
        pages: list[ConfluencePage] = []
        start = 0
        page_size = min(limit, 50)

        while len(pages) < limit:
            payload = await self._request_json(
                "GET",
                "/content/search",
                params={
                    "cql": cql,
                    "expand": "body.storage,version,space",
                    "limit": page_size,
                    "start": start,
                },
            )
            results = payload.get("results", [])
            if not results:
                break

            for result in results:
                pages.append(self._to_page(result))
                if len(pages) >= limit:
                    break

            if len(results) < page_size:
                break
            start += len(results)

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
        since_text = since.strftime("%Y-%m-%d %H:%M")
        space_key = self._quote_cql_value(self.settings.confluence_space_key)
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
