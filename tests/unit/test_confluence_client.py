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
    def __init__(self, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
        # list 를 주면 요청 순서대로 하나씩 반환 (커서 페이지네이션 테스트용)
        self.payloads = payload if isinstance(payload, list) else [payload]
        self.requests: list[dict[str, Any]] = []

    async def request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> FakeResponse:
        self.requests.append({"method": method, "url": url, "params": params, "json": json})
        index = min(len(self.requests) - 1, len(self.payloads) - 1)
        return FakeResponse(self.payloads[index])


def _settings() -> Settings:
    return Settings(
        confluence_base_url="https://example.atlassian.net",
        confluence_user_email="user@example.com",
        confluence_api_token="token",
        confluence_space_key="OnRamp",
        confluence_timezone="Asia/Seoul",
    )


def test_build_recent_pages_cql_uses_space_and_since_time() -> None:
    client = ConfluenceClient(settings=_settings(), client=FakeAsyncClient({"results": []}))  # type: ignore[arg-type]

    cql = client._build_recent_pages_cql(datetime(2026, 5, 28, 12, 30))

    assert cql == 'type = page AND space = "OnRamp" AND lastmodified >= "2026-05-28 12:30" ORDER BY lastmodified DESC'


async def test_fetch_recent_pages_reads_storage_body() -> None:
    payload = {
        "results": [
            {
                "id": "123",
                "title": "API Runbook",
                "space": {"key": "OnRamp"},
                "body": {"storage": {"value": "<main><h1>API Runbook</h1></main>"}},
                "version": {"when": "2026-05-28T12:35:00.000+0900", "number": 7},
                "_links": {"webui": "/spaces/OnRamp/pages/123/API+Runbook"},
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
    assert pages[0].url == "https://example.atlassian.net/wiki/spaces/OnRamp/pages/123/API+Runbook"
    assert fake_client.requests[0]["url"] == "https://example.atlassian.net/wiki/rest/api/content/search"
    assert fake_client.requests[0]["params"]["expand"] == "body.storage,version,space,metadata.labels"
    assert pages[0].labels == ()  # metadata 부재 → 빈 튜플 (방어)


async def test_fetch_pages_extracts_labels() -> None:
    payload = {
        "results": [
            {
                "id": "124",
                "title": "Content Negotiation [a78792-639072]",
                "space": {"key": "OnRamp"},
                "body": {"storage": {"value": "<p>doc</p>"}},
                "version": {"when": "2026-05-25T10:00:00.000+0900", "number": 1},
                "metadata": {
                    "labels": {
                        "results": [
                            {"name": "auto-imported"},
                            {"name": "site-apache"},
                            {"name": "version-2-4"},
                            {"name": ""},  # 빈 이름은 제외
                        ]
                    }
                },
                "_links": {"webui": "/spaces/OnRamp/pages/124/CN"},
            }
        ]
    }
    client = ConfluenceClient(settings=_settings(), client=FakeAsyncClient(payload))  # type: ignore[arg-type]

    pages = await client.fetch_recent_pages(hours=1, limit=10)

    assert pages[0].labels == ("auto-imported", "site-apache", "version-2-4")


async def test_update_page_writes_next_version() -> None:
    fake_client = FakeAsyncClient({})
    client = ConfluenceClient(settings=_settings(), client=fake_client)  # type: ignore[arg-type]
    page = ConfluencePage(
        page_id="123",
        title="API Runbook",
        space_key="OnRamp",
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


async def test_fetch_candidate_pages_uses_title_order_not_recent_order() -> None:
    fake_client = FakeAsyncClient({"results": []})
    client = ConfluenceClient(settings=_settings(), client=fake_client)  # type: ignore[arg-type]

    await client.fetch_candidate_pages(limit=10)

    cql = fake_client.requests[0]["params"]["cql"]
    assert cql == 'type = page AND space = "OnRamp" ORDER BY title ASC, id ASC'


async def test_fetch_all_pages_omits_lastmodified() -> None:
    fake_client = FakeAsyncClient({"results": []})
    client = ConfluenceClient(settings=_settings(), client=fake_client)  # type: ignore[arg-type]

    await client.fetch_all_pages(limit=10)

    cql = fake_client.requests[0]["params"]["cql"]
    assert cql == 'type = page AND space = "OnRamp" ORDER BY title ASC, id ASC'
    assert "lastmodified" not in cql


def _search_result(page_id: str) -> dict[str, Any]:
    return {
        "id": page_id,
        "title": f"Page {page_id}",
        "space": {"key": "OnRamp"},
        "body": {"storage": {"value": f"<p>{page_id}</p>"}},
        "version": {"when": "2026-06-10T00:00:00.000+0900", "number": 1},
        "_links": {"webui": f"/spaces/OnRamp/pages/{page_id}"},
    }


async def test_search_follows_next_cursor_links() -> None:
    """Cloud가 start 오프셋을 무시하므로 _links.next 커서로만 다음 페이지를 가져온다."""
    fake_client = FakeAsyncClient(
        [
            {
                "results": [_search_result("1"), _search_result("2")],
                "_links": {"next": "/rest/api/content/search?cursor=abc&cql=ignored&limit=2"},
            },
            {"results": [_search_result("3")]},  # next 없음 → 종료
        ]
    )
    client = ConfluenceClient(settings=_settings(), client=fake_client)  # type: ignore[arg-type]

    pages = await client.fetch_all_pages(limit=10)

    assert [p.page_id for p in pages] == ["1", "2", "3"]
    assert len(fake_client.requests) == 2
    # 첫 요청은 start 없이 cql 로, 두 번째 요청은 next 링크의 쿼리(cursor)로
    assert "start" not in fake_client.requests[0]["params"]
    assert fake_client.requests[1]["params"]["cursor"] == "abc"


async def test_search_stops_at_limit_mid_page() -> None:
    fake_client = FakeAsyncClient(
        [
            {
                "results": [_search_result("1"), _search_result("2"), _search_result("3")],
                "_links": {"next": "/rest/api/content/search?cursor=abc"},
            },
        ]
    )
    client = ConfluenceClient(settings=_settings(), client=fake_client)  # type: ignore[arg-type]

    pages = await client.fetch_all_pages(limit=2)

    assert [p.page_id for p in pages] == ["1", "2"]
    assert len(fake_client.requests) == 1  # limit 도달 → next 따라가지 않음


async def test_search_dedupes_repeated_pages_from_server() -> None:
    """서버가 같은 결과를 반복 반환해도(과거 start 무시 증상) 중복 수집하지 않는다."""
    duplicated = {
        "results": [_search_result("1"), _search_result("2")],
        "_links": {"next": "/rest/api/content/search?cursor=abc"},
    }
    fake_client = FakeAsyncClient([duplicated, {"results": [_search_result("1"), _search_result("2")]}])
    client = ConfluenceClient(settings=_settings(), client=fake_client)  # type: ignore[arg-type]

    pages = await client.fetch_all_pages(limit=10)

    assert [p.page_id for p in pages] == ["1", "2"]


async def test_search_breaks_when_cursor_stops_progressing() -> None:
    """next 커서가 계속 와도 신규 페이지가 0건이면 중단한다 (무한 루프 방지)."""
    stuck_payload = {
        "results": [_search_result("1"), _search_result("2")],
        "_links": {"next": "/rest/api/content/search?cursor=stuck"},
    }
    fake_client = FakeAsyncClient([stuck_payload])  # 같은 응답을 무한 반복하는 서버
    client = ConfluenceClient(settings=_settings(), client=fake_client)  # type: ignore[arg-type]

    pages = await client.fetch_all_pages(limit=10)

    assert [p.page_id for p in pages] == ["1", "2"]
    assert len(fake_client.requests) == 2  # 1회차 수집 + 2회차 전부 중복 → 즉시 중단


async def test_create_page_posts_storage_payload() -> None:
    fake_client = FakeAsyncClient({"id": "999", "title": "보고서", "_links": {"webui": "/spaces/OnRamp/pages/999"}})
    client = ConfluenceClient(settings=_settings(), client=fake_client)  # type: ignore[arg-type]

    page = await client.create_page(title="보고서", html="<h2>현재 상황</h2><p>내용</p>")

    request = fake_client.requests[0]
    assert request["method"] == "POST"
    assert request["url"] == "https://example.atlassian.net/wiki/rest/api/content"
    body = request["json"]
    assert body["type"] == "page"
    assert body["title"] == "보고서"
    assert body["space"]["key"] == "OnRamp"
    assert body["body"]["storage"]["value"] == "<h2>현재 상황</h2><p>내용</p>"
    assert body["body"]["storage"]["representation"] == "storage"
    assert page.page_id == "999"
