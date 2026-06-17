from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

import app.api.v1.auth_browser as ab
from app.auth.session import issue_session_token
from app.config import Settings
from app.middleware.error_handler import register_error_handlers
from app.models.auth import AuthSessionResponse, SlackAuthorizeResponse

SECRET = "x" * 40


def _build(monkeypatch):
    settings = Settings(
        _env_file=None,
        debug=False,
        auth_jwt_secret=SecretStr(SECRET),
        auth_cookie_secure=False,
        frontend_post_login_redirect="/chat",
    )
    monkeypatch.setattr("app.api.v1.auth_browser.get_settings", lambda: settings)
    monkeypatch.setattr("app.auth.session.get_settings", lambda: settings)
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(ab.browser_router)
    return TestClient(app), settings


def test_login_redirects_to_slack(monkeypatch):
    client, _ = _build(monkeypatch)
    monkeypatch.setattr(
        ab,
        "build_slack_authorization",
        lambda settings, team=None: SlackAuthorizeResponse(
            authorization_url="https://slack.com/openid/connect/authorize?client_id=x",
            state_expires_at=datetime.now(UTC) + timedelta(minutes=5),
        ),
    )
    r = client.get("/auth/login", follow_redirects=False)
    assert r.status_code == 307
    assert "slack.com/openid/connect/authorize" in r.headers["location"]


def test_callback_sets_cookie_and_redirects_to_frontend(monkeypatch):
    client, _ = _build(monkeypatch)

    async def fake_auth(*, code, state, settings):
        return AuthSessionResponse(
            access_token="ignored",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            tenant_id="tenant1-onramp",
            external_tenant="T089ENT4A2D",
            user_id="U1",
            name="양정우",
            email="a@b.com",
        )

    monkeypatch.setattr(ab, "authenticate_with_slack_callback", fake_auth)
    r = client.get("/auth/callback?code=abc&state=xyz", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/chat"
    assert "onramp_session=" in r.headers.get("set-cookie", "")


def test_callback_missing_code_returns_400(monkeypatch):
    client, _ = _build(monkeypatch)
    r = client.get("/auth/callback", follow_redirects=False)
    assert r.status_code == 400


def test_me_returns_user_from_cookie(monkeypatch):
    client, settings = _build(monkeypatch)
    token, _ = issue_session_token(
        tenant_id="tenant1-onramp", subject="U1", settings=settings, provider="slack", name="양정우", email="a@b.com"
    )
    client.cookies.set("onramp_session", token)
    r = client.get("/auth/me")
    assert r.status_code == 200
    body = r.json()
    assert body["tenant_id"] == "tenant1-onramp"
    assert body["subject"] == "U1"
    assert body["name"] == "양정우"


def test_me_without_cookie_401(monkeypatch):
    client, _ = _build(monkeypatch)
    r = client.get("/auth/me")
    assert r.status_code == 401


def test_logout_returns_204(monkeypatch):
    client, _ = _build(monkeypatch)
    r = client.post("/auth/logout")
    assert r.status_code == 204
