from __future__ import annotations

from datetime import datetime
from typing import Any

from app.config import Settings
from app.db.confluence import ConfluenceClient, ConfluencePage


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeAsyncClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.requests: list[dict[str, Any]] = []

    async def request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> FakeResponse:
        self.requests.append({"method": method, "url": url, "params": params, "json": json})
        return FakeResponse(self.payload)


def _settings() -> Settings:
    return Settings(
        confluence_base_url="https://example.atlassian.net",
        confluence_user_email="user@example.com",
        confluence_api_token="token",
        confluence_space_key="TRUSTRAG",
        confluence_timezone="Asia/Seoul",
    )


def test_build_recent_pages_cql_uses_space_and_since_time() -> None:
    client = ConfluenceClient(settings=_settings(), client=FakeAsyncClient({"results": []}))  # type: ignore[arg-type]

    cql = client._build_recent_pages_cql(datetime(2026, 5, 28, 12, 30))

    assert cql == 'type = page AND space = "TRUSTRAG" AND lastmodified >= "2026-05-28 12:30" ORDER BY lastmodified DESC'


async def test_fetch_recent_pages_reads_storage_body() -> None:
    payload = {
        "results": [
            {
                "id": "123",
                "title": "API Runbook",
                "space": {"key": "TRUSTRAG"},
                "body": {"storage": {"value": "<main><h1>API Runbook</h1></main>"}},
                "version": {"when": "2026-05-28T12:35:00.000+0900", "number": 7},
                "_links": {"webui": "/spaces/TRUSTRAG/pages/123/API+Runbook"},
            }
        ]
    }
    fake_client = FakeAsyncClient(payload)
    client = ConfluenceClient(settings=_settings(), client=fake_client)  # type: ignore[arg-type]

    pages = await client.fetch_recent_pages(hours=1, limit=10)

    assert len(pages) == 1
    assert pages[0].page_id == "123"
    assert pages[0].title == "API Runbook"
    assert pages[0].html == "<main><h1>API Runbook</h1></main>"
    assert pages[0].version == 7
    assert pages[0].url == "https://example.atlassian.net/wiki/spaces/TRUSTRAG/pages/123/API+Runbook"
    assert fake_client.requests[0]["url"] == "https://example.atlassian.net/wiki/rest/api/content/search"
    assert fake_client.requests[0]["params"]["expand"] == "body.storage,version,space"


async def test_update_page_writes_next_version() -> None:
    fake_client = FakeAsyncClient({})
    client = ConfluenceClient(settings=_settings(), client=fake_client)  # type: ignore[arg-type]
    page = ConfluencePage(
        page_id="123",
        title="API Runbook",
        space_key="TRUSTRAG",
        html="<h1>Old</h1>",
        last_modified="2026-05-28T12:35:00.000+0900",
        version=7,
        url="",
    )

    await client.update_page(page, "<h1>New</h1>", next_version=8)

    request = fake_client.requests[0]
    assert request["method"] == "PUT"
    assert request["url"] == "https://example.atlassian.net/wiki/rest/api/content/123"
    assert request["json"]["version"]["number"] == 8
    assert request["json"]["body"]["storage"]["value"] == "<h1>New</h1>"
