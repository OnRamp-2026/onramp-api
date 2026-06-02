"""POST/GET/PATCH /v1/asset 통합 테스트 (HITL).

ASSET LLM과 Confluence 페이지 생성을 stub해 네트워크 없이 전체 흐름을 검증한다.
"""

import json

import pytest

from app.db.confluence import ConfluencePage


def _asset_resp(title: str = "장애 회의 보고서") -> str:
    """ASSET LLM 응답(5요소 JSON)을 만든다."""
    return json.dumps(
        {
            "title": title,
            "situation": "결제 서비스 지연 발생",
            "cause": "DB 커넥션 풀 고갈",
            "evidence": "회의에서 풀 사이즈 부족 확인",
            "solution": "풀 사이즈 상향\n모니터링 추가",
            "infra_context": "RDS, 커넥션 풀 50",
        }
    )


@pytest.fixture(autouse=True)
def _clear_draft_store():
    """프로세스 전역 _draft_store를 테스트마다 비워 격리."""
    from app.services import asset_service

    asset_service._draft_store.clear()
    yield
    asset_service._draft_store.clear()


@pytest.fixture
def stub_asset(monkeypatch):
    """ASSET LLM + Confluence create_page를 stub."""

    async def _llm(*args, **kwargs):
        return _asset_resp()

    async def _create_page(self, title, html, space_key=None):
        return ConfluencePage(
            page_id="p1",
            title=title,
            space_key="OnRamp",
            html=html,
            last_modified="",
            version=1,
            url="https://team3exampledoc.atlassian.net/",
        )

    monkeypatch.setattr("app.services.asset_service.call_llm", _llm)
    monkeypatch.setattr("app.db.confluence.ConfluenceClient.create_page", _create_page)
    return monkeypatch


@pytest.mark.asyncio
async def test_create_report(client, stub_asset):
    """녹취 → 5요소 초안 생성 (status=draft)."""
    resp = await client.post("/v1/asset", json={"transcript": "장애 대응 회의 녹취 내용...", "category": "장애대응"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "draft"
    assert data["report"]["situation"] != ""
    assert data["report_id"]


@pytest.mark.asyncio
async def test_create_report_auto_title(client, stub_asset):
    """title 미지정 시 LLM이 생성한 제목을 사용한다."""
    resp = await client.post("/v1/asset", json={"transcript": "회의 녹취 텍스트 충분히 김"})
    assert resp.status_code == 200
    assert resp.json()["title"] == "장애 회의 보고서"  # stub LLM이 반환한 제목


@pytest.mark.asyncio
async def test_create_report_short_transcript(client):
    """transcript 10자 미만 → 422."""
    resp = await client.post("/v1/asset", json={"transcript": "짧음"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_get_report(client, stub_asset):
    """초안 생성 후 조회."""
    created = (await client.post("/v1/asset", json={"transcript": "회의 녹취 텍스트입니다"})).json()
    resp = await client.get(f"/v1/asset/{created['report_id']}")
    assert resp.status_code == 200
    assert resp.json()["report_id"] == created["report_id"]


@pytest.mark.asyncio
async def test_get_report_not_found(client):
    """없는 id 조회 → 404."""
    resp = await client.get("/v1/asset/does-not-exist")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_report_partial(client, stub_asset):
    """PATCH로 보낸 필드만 수정, 나머지 유지."""
    created = (await client.post("/v1/asset", json={"transcript": "회의 녹취 텍스트입니다"})).json()
    rid = created["report_id"]
    resp = await client.patch(f"/v1/asset/{rid}", json={"title": "수정된 제목", "situation": "수정된 상황"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "수정된 제목"
    assert data["report"]["situation"] == "수정된 상황"
    assert data["report"]["cause"] == created["report"]["cause"]  # 미수정 필드 유지


@pytest.mark.asyncio
async def test_update_report_not_found(client):
    """없는 id 수정 → 404."""
    resp = await client.patch("/v1/asset/does-not-exist", json={"title": "x"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_full_hitl_flow(client, stub_asset):
    """생성 → 수정 → 승인 → Confluence 등록(수정본)."""
    rid = (await client.post("/v1/asset", json={"transcript": "회의 녹취 텍스트입니다"})).json()["report_id"]
    await client.patch(f"/v1/asset/{rid}", json={"situation": "최종 수정 상황"})
    resp = await client.post(f"/v1/asset/{rid}/approve")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "published"
    assert data["confluence_url"] != ""
    # 승인 후 조회 시 published + url 반영
    after = (await client.get(f"/v1/asset/{rid}")).json()
    assert after["status"] == "published"
    assert after["report"]["situation"] == "최종 수정 상황"


@pytest.mark.asyncio
async def test_approve_not_found(client):
    """없는 id 승인 → 404."""
    resp = await client.post("/v1/asset/does-not-exist/approve")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_patch_after_publish_blocked(client, stub_asset):
    """published 이후 PATCH는 409 (Confluence와 불일치 방지)."""
    rid = (await client.post("/v1/asset", json={"transcript": "회의 녹취 텍스트입니다"})).json()["report_id"]
    await client.post(f"/v1/asset/{rid}/approve")
    resp = await client.patch(f"/v1/asset/{rid}", json={"title": "다시 수정"})
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_reapprove_blocked(client, stub_asset):
    """이미 published면 재승인 409 (중복 페이지 방지)."""
    rid = (await client.post("/v1/asset", json={"transcript": "회의 녹취 텍스트입니다"})).json()["report_id"]
    await client.post(f"/v1/asset/{rid}/approve")
    resp = await client.post(f"/v1/asset/{rid}/approve")
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_approve_confluence_failure_502(client, stub_asset, monkeypatch):
    """Confluence 등록 실패는 일반 500이 아니라 502로 변환."""
    rid = (await client.post("/v1/asset", json={"transcript": "회의 녹취 텍스트입니다"})).json()["report_id"]

    async def _boom(self, title, html, space_key=None):
        raise RuntimeError("Confluence 5xx")

    monkeypatch.setattr("app.db.confluence.ConfluenceClient.create_page", _boom)
    resp = await client.post(f"/v1/asset/{rid}/approve")
    assert resp.status_code == 502


@pytest.mark.asyncio
async def test_create_empty_report_502(client, monkeypatch):
    """LLM이 형식만 맞고 내용이 빈 응답 → 502 (빈 보고서 저장 방지)."""

    async def _empty_llm(*args, **kwargs):
        return json.dumps(
            {"title": "", "situation": "", "cause": "", "evidence": "", "solution": "", "infra_context": ""}
        )

    monkeypatch.setattr("app.services.asset_service.call_llm", _empty_llm)
    resp = await client.post("/v1/asset", json={"transcript": "회의 녹취 텍스트입니다"})
    assert resp.status_code == 502
