from __future__ import annotations

from typing import Any

from app.config import Settings
from app.db.confluence import ConfluencePage
from scripts import random_confluence_page_editor
from scripts.random_confluence_page_editor import _upsert_test_section, update_random_pages


def test_upsert_test_section_replaces_existing_section() -> None:
    first = _upsert_test_section("<h1>Doc</h1>", "Doc", "2026-05-28 15:00 KST")
    second = _upsert_test_section(first, "Doc", "2026-05-28 15:30 KST")

    assert second.count("ONRAMP_TEST_SECTION_START") == 1
    assert "2026-05-28 15:30 KST" in second
    assert "2026-05-28 15:00 KST" not in second
    assert "breakoutWidth" in second
    assert "kubectl get pod <pod-name>" in second


def test_upsert_test_section_appends_when_no_section_exists() -> None:
    existing_html = "<h1>Doc</h1><p>Content</p>"

    result = _upsert_test_section(existing_html, "Doc", "2026-05-28 15:00 KST")

    assert existing_html in result
    assert result.count("ONRAMP_TEST_SECTION_START") == 1
    assert "2026-05-28 15:00 KST" in result


def test_upsert_test_section_escapes_html_inputs() -> None:
    result = _upsert_test_section(
        "<h1>Doc</h1>",
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert(1)>",
    )

    assert "<script>" not in result
    assert "<img src=" not in result
    assert "&lt;script&gt;" in result
    assert "&lt;img" in result


class FakeConfluenceClient:
    pages: list[ConfluencePage] = []
    fail_page_ids: set[str] = set()
    updates: list[tuple[str, int]] = []

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def fetch_candidate_pages(self, limit: int) -> list[ConfluencePage]:
        return self.pages[:limit]

    async def update_page(self, page: ConfluencePage, updated_html: str, next_version: int) -> None:
        if page.page_id in self.fail_page_ids:
            raise RuntimeError("boom")
        self.updates.append((page.page_id, next_version))


def _settings() -> Settings:
    return Settings(
        confluence_base_url="https://example.atlassian.net",
        confluence_user_email="user@example.com",
        confluence_api_token="token",
        confluence_space_key="OnRamp",
        confluence_timezone="Asia/Seoul",
    )


def _page(page_id: str, html: str = "<h1>Doc</h1>", version: int | None = 1) -> ConfluencePage:
    return ConfluencePage(
        page_id=page_id,
        title=f"Doc {page_id}",
        space_key="OnRamp",
        html=html,
        last_modified="",
        version=version,
        url="",
    )


async def test_update_random_pages_samples_from_all_candidate_pages(monkeypatch: Any) -> None:
    FakeConfluenceClient.pages = [_page("one"), _page("two"), _page("three")]
    FakeConfluenceClient.fail_page_ids = set()
    FakeConfluenceClient.updates = []
    monkeypatch.setattr(random_confluence_page_editor, "get_settings", _settings)
    monkeypatch.setattr(random_confluence_page_editor, "ConfluenceClient", FakeConfluenceClient)

    previews = await update_random_pages(count=2, candidate_limit=10, seed=1, apply=True)

    assert len(previews) == 2
    assert {preview.page_id for preview in previews}.issubset({"one", "two", "three"})
    assert len(FakeConfluenceClient.updates) == 2


async def test_update_random_pages_continues_when_one_page_update_fails(monkeypatch: Any) -> None:
    FakeConfluenceClient.pages = [_page("fail"), _page("ok")]
    FakeConfluenceClient.fail_page_ids = {"fail"}
    FakeConfluenceClient.updates = []
    monkeypatch.setattr(random_confluence_page_editor, "get_settings", _settings)
    monkeypatch.setattr(random_confluence_page_editor, "ConfluenceClient", FakeConfluenceClient)

    previews = await update_random_pages(count=2, candidate_limit=10, seed=0, apply=True)

    assert [preview.page_id for preview in previews] == ["ok"]
    assert FakeConfluenceClient.updates == [("ok", 2)]
