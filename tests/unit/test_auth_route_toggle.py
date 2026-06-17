import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def auth_route_app_factory(monkeypatch: pytest.MonkeyPatch):
    def factory(*, enabled: bool):
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv("AUTH_ENABLE_SLACK_LOGIN", "true" if enabled else "false")

        from app.config import get_settings

        get_settings.cache_clear()
        from app.main import create_app

        return create_app()

    yield factory

    from app.config import get_settings

    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_slack_auth_routes_are_hidden_when_disabled(auth_route_app_factory) -> None:
    transport = ASGITransport(app=auth_route_app_factory(enabled=False))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/auth/slack/authorize")
        public_callback = await client.get("/auth/callback")

    assert response.status_code == 404
    assert public_callback.status_code == 404


@pytest.mark.asyncio
async def test_slack_auth_routes_are_exposed_only_when_enabled(auth_route_app_factory) -> None:
    transport = ASGITransport(app=auth_route_app_factory(enabled=True))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/auth/slack/authorize")
        public_callback = await client.get("/auth/callback")

    assert response.status_code == 503
    assert response.json()["error"] == "Slack 로그인 설정이 구성되지 않았습니다."
    assert public_callback.status_code == 400
    assert public_callback.json()["error"] == "Slack callback에 code/state가 없습니다."
