from __future__ import annotations

from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings

JWT_SECRET = "test-auth-secret-with-at-least-32-bytes"
SLACK_CLIENT_ID = "111.222"
SLACK_CLIENT_SECRET = "slack-client-secret"
SLACK_REDIRECT_URI = "https://shared.example.com/v1/auth/slack/callback"


def _settings(*, registry: dict[str, object] | None = None) -> Settings:
    return Settings(
        _env_file=None,
        debug=False,
        auth_jwt_secret=JWT_SECRET,
        auth_jwt_issuer="shared-onramp-auth",
        auth_jwt_audience="onramp-api",
        auth_slack_client_id=SLACK_CLIENT_ID,
        auth_slack_client_secret=SLACK_CLIENT_SECRET,
        auth_slack_redirect_uri=SLACK_REDIRECT_URI,
        tenant_registry={"slack:T12345": "tenant1-onramp"} if registry is None else registry,
    )


def _slack_id_token(*, nonce: str, team_id: str = "T12345", user_id: str = "U12345") -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "iss": "https://slack.com",
            "sub": user_id,
            "aud": SLACK_CLIENT_ID,
            "exp": int((now + timedelta(minutes=5)).timestamp()),
            "iat": int(now.timestamp()),
            "nonce": nonce,
            "https://slack.com/team_id": team_id,
            "https://slack.com/user_id": user_id,
            "email": "alice@example.com",
            "email_verified": True,
            "name": "Alice",
        },
        "unused-test-secret-with-at-least-32-bytes",
        algorithm="HS256",
    )


@pytest.fixture
async def slack_auth_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("AUTH_ENABLE_SLACK_LOGIN", "true")

    from app.config import get_settings

    get_settings.cache_clear()
    from app.main import create_app

    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_slack_authorize_returns_signed_state_url(
    slack_auth_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.api.v1.auth.get_settings", lambda: _settings())

    response = await slack_auth_client.get("/v1/auth/slack/authorize", params={"team": "T12345"})

    assert response.status_code == 200
    data = response.json()
    parsed = urlparse(data["authorization_url"])
    query = parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "slack.com"
    assert parsed.path == "/openid/connect/authorize"
    assert query["client_id"] == [SLACK_CLIENT_ID]
    assert query["scope"] == ["openid profile email"]
    assert query["redirect_uri"] == [SLACK_REDIRECT_URI]
    assert query["team"] == ["T12345"]
    assert "state" in query
    assert "nonce" in query


@pytest.mark.asyncio
async def test_slack_callback_issues_internal_session_token(
    slack_auth_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    monkeypatch.setattr("app.api.v1.auth.get_settings", lambda: settings)

    authorize = await slack_auth_client.get("/v1/auth/slack/authorize")
    query = parse_qs(urlparse(authorize.json()["authorization_url"]).query)
    state = query["state"][0]
    nonce = query["nonce"][0]

    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://slack.com/api/openid.connect.token"
        # Slack OAuth 엔드포인트는 form-encoded만 파싱한다(JSON 바디면 invalid_code).
        assert request.headers["content-type"] == "application/x-www-form-urlencoded"
        body = request.read().decode()
        assert "grant_type=authorization_code" in body
        assert "code=test-code" in body
        return httpx.Response(
            200,
            request=request,
            json={
                "ok": True,
                "access_token": "xoxp-test",
                "token_type": "Bearer",
                "id_token": _slack_id_token(nonce=nonce),
            },
        )

    class MockAsyncClient(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("app.auth.slack_oidc.httpx.AsyncClient", MockAsyncClient)

    response = await slack_auth_client.get(
        "/v1/auth/slack/callback",
        params={"code": "test-code", "state": state},
    )

    assert response.status_code == 200
    data = response.json()
    claims = jwt.decode(
        data["access_token"],
        JWT_SECRET,
        algorithms=["HS256"],
        audience="onramp-api",
        issuer="shared-onramp-auth",
    )
    assert data["tenant_id"] == "tenant1-onramp"
    assert data["tenant_api_base_url"] == ""
    assert data["provider"] == "slack"
    assert data["external_tenant"] == "T12345"
    assert data["user_id"] == "U12345"
    assert data["email"] == "alice@example.com"
    assert claims["tenant_id"] == "tenant1-onramp"
    assert claims["provider"] == "slack"
    assert claims["external_tenant"] == "T12345"
    assert claims["sub"] == "U12345"


@pytest.mark.asyncio
async def test_public_auth_callback_sets_cookie_and_redirects(
    slack_auth_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # /auth/callback = 브라우저 SPA용(쿠키 세션 + 화면 redirect). JSON 토큰 콜백은 /v1/auth/slack/callback.
    settings = _settings()
    monkeypatch.setattr("app.api.v1.auth.get_settings", lambda: settings)
    monkeypatch.setattr("app.api.v1.auth_browser.get_settings", lambda: settings)

    authorize = await slack_auth_client.get("/v1/auth/slack/authorize")
    query = parse_qs(urlparse(authorize.json()["authorization_url"]).query)
    state = query["state"][0]
    nonce = query["nonce"][0]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "ok": True,
                "access_token": "xoxp-test",
                "token_type": "Bearer",
                "id_token": _slack_id_token(nonce=nonce),
            },
        )

    class MockAsyncClient(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("app.auth.slack_oidc.httpx.AsyncClient", MockAsyncClient)

    response = await slack_auth_client.get(
        "/auth/callback",
        params={"code": "test-code", "state": state},
    )

    # 브라우저 콜백: 세션 쿠키 set + 프론트로 redirect (JSON 본문 아님)
    assert response.status_code == 303
    assert "onramp_session=" in response.headers.get("set-cookie", "")


@pytest.mark.asyncio
async def test_slack_callback_rejects_invalid_state(
    slack_auth_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.api.v1.auth.get_settings", lambda: _settings())

    response = await slack_auth_client.get(
        "/v1/auth/slack/callback",
        params={"code": "test-code", "state": "bad-state"},
    )

    assert response.status_code == 401
    assert "state" in response.json()["error"]


@pytest.mark.asyncio
async def test_slack_callback_rejects_unregistered_workspace(
    slack_auth_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(registry={})
    monkeypatch.setattr("app.api.v1.auth.get_settings", lambda: settings)

    authorize = await slack_auth_client.get("/v1/auth/slack/authorize")
    query = parse_qs(urlparse(authorize.json()["authorization_url"]).query)
    state = query["state"][0]
    nonce = query["nonce"][0]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "ok": True,
                "access_token": "xoxp-test",
                "token_type": "Bearer",
                "id_token": _slack_id_token(nonce=nonce),
            },
        )

    class MockAsyncClient(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("app.auth.slack_oidc.httpx.AsyncClient", MockAsyncClient)

    response = await slack_auth_client.get(
        "/v1/auth/slack/callback",
        params={"code": "test-code", "state": state},
    )

    assert response.status_code == 403
    assert "등록되지 않은 Slack workspace" in response.json()["error"]


@pytest.mark.asyncio
async def test_slack_callback_returns_tenant_api_base_url_when_registered(
    slack_auth_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        registry={
            "slack:T12345": {
                "tenant_id": "tenant1-onramp",
                "tenant_api_base_url": "https://tenant1.example.com",
            }
        }
    )
    monkeypatch.setattr("app.api.v1.auth.get_settings", lambda: settings)

    authorize = await slack_auth_client.get("/v1/auth/slack/authorize")
    query = parse_qs(urlparse(authorize.json()["authorization_url"]).query)
    state = query["state"][0]
    nonce = query["nonce"][0]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "ok": True,
                "access_token": "xoxp-test",
                "token_type": "Bearer",
                "id_token": _slack_id_token(nonce=nonce),
            },
        )

    class MockAsyncClient(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("app.auth.slack_oidc.httpx.AsyncClient", MockAsyncClient)

    response = await slack_auth_client.get(
        "/v1/auth/slack/callback",
        params={"code": "test-code", "state": state},
    )

    assert response.status_code == 200
    assert response.json()["tenant_api_base_url"] == "https://tenant1.example.com"


@pytest.mark.asyncio
async def test_slack_callback_accepts_allowed_host(
    slack_auth_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        registry={
            "slack:T12345": {
                "tenant_id": "tenant1-onramp",
                "tenant_api_base_url": "https://tenant1.example.com",
                "allowed_hosts": ["test"],
            }
        }
    )
    monkeypatch.setattr("app.api.v1.auth.get_settings", lambda: settings)

    authorize = await slack_auth_client.get("/v1/auth/slack/authorize")
    query = parse_qs(urlparse(authorize.json()["authorization_url"]).query)
    state = query["state"][0]
    nonce = query["nonce"][0]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "ok": True,
                "access_token": "xoxp-test",
                "token_type": "Bearer",
                "id_token": _slack_id_token(nonce=nonce),
            },
        )

    class MockAsyncClient(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("app.auth.slack_oidc.httpx.AsyncClient", MockAsyncClient)

    response = await slack_auth_client.get(
        "/v1/auth/slack/callback",
        params={"code": "test-code", "state": state},
    )

    assert response.status_code == 200
    assert response.json()["tenant_id"] == "tenant1-onramp"


@pytest.mark.asyncio
async def test_slack_callback_rejects_host_mismatch(
    slack_auth_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        registry={
            "slack:T12345": {
                "tenant_id": "tenant1-onramp",
                "tenant_api_base_url": "https://tenant1.example.com",
                "allowed_hosts": ["tenant1.example.com"],
            }
        }
    )
    monkeypatch.setattr("app.api.v1.auth.get_settings", lambda: settings)

    authorize = await slack_auth_client.get("/v1/auth/slack/authorize")
    query = parse_qs(urlparse(authorize.json()["authorization_url"]).query)
    state = query["state"][0]
    nonce = query["nonce"][0]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "ok": True,
                "access_token": "xoxp-test",
                "token_type": "Bearer",
                "id_token": _slack_id_token(nonce=nonce),
            },
        )

    class MockAsyncClient(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("app.auth.slack_oidc.httpx.AsyncClient", MockAsyncClient)

    response = await slack_auth_client.get(
        "/v1/auth/slack/callback",
        params={"code": "test-code", "state": state},
    )

    assert response.status_code == 403
    assert "host" in response.json()["error"]
