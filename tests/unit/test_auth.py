from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials

from app.api.deps import decode_tenant_token, get_current_tenant
from app.config import Settings

SECRET = "test-auth-secret-with-at-least-32-bytes"
COOKIE_NAME = Settings().auth_cookie_name


def _token(**claims: object) -> str:
    payload = {
        "tenant_id": "tenant-a",
        "aud": "onramp-api",
        "exp": datetime.now(UTC) + timedelta(minutes=5),
        **claims,
    }
    return jwt.encode(payload, SECRET, algorithm="HS256")


def _request(
    *,
    method: str = "GET",
    cookie: str | None = None,
    origin: str | None = None,
    cookie_name: str = COOKIE_NAME,
) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if cookie:
        headers.append((b"cookie", f"{cookie_name}={cookie}".encode()))
    if origin:
        headers.append((b"origin", origin.encode()))
    return Request(
        {
            "type": "http",
            "method": method,
            "path": "/v1/transcriptions",
            "headers": headers,
        }
    )


def test_decode_tenant_token_uses_verified_claim() -> None:
    settings = Settings(auth_jwt_secret=SECRET)

    assert decode_tenant_token(_token(), settings) == "tenant-a"


@pytest.mark.parametrize(
    "token",
    [
        jwt.encode(
            {"tenant_id": "tenant-a", "aud": "onramp-api", "exp": datetime.now(UTC) - timedelta(seconds=1)},
            SECRET,
            algorithm="HS256",
        ),
        jwt.encode(
            {"tenant_id": "tenant-a", "aud": "onramp-api", "exp": datetime.now(UTC) + timedelta(minutes=5)},
            "wrong-auth-secret-with-at-least-32-bytes",
            algorithm="HS256",
        ),
        _token(tenant_id="tenant/a"),
    ],
)
def test_decode_tenant_token_rejects_invalid_tokens(token: str) -> None:
    settings = Settings(auth_jwt_secret=SECRET)

    with pytest.raises(HTTPException) as exc_info:
        decode_tenant_token(token, settings)

    assert exc_info.value.status_code == 401


def test_decode_tenant_token_requires_auth_configuration() -> None:
    with pytest.raises(HTTPException) as exc_info:
        decode_tenant_token(_token(), Settings(auth_jwt_secret=""))

    assert exc_info.value.status_code == 503


def test_get_current_tenant_requires_cookie_or_bearer() -> None:
    with pytest.raises(HTTPException) as exc_info:
        get_current_tenant(_request(), None)

    assert exc_info.value.status_code == 401


def test_get_current_tenant_accepts_bearer_without_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(auth_jwt_secret=SECRET)
    monkeypatch.setattr("app.api.deps.get_settings", lambda: settings)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=_token())

    assert get_current_tenant(_request(method="POST"), credentials) == "tenant-a"


def test_get_current_tenant_accepts_session_cookie(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        auth_jwt_secret=SECRET,
        auth_base_url="https://onramp.example.com",
    )
    monkeypatch.setattr("app.api.deps.get_settings", lambda: settings)

    assert (
        get_current_tenant(
            _request(
                method="POST",
                cookie=_token(),
                origin="https://onramp.example.com",
            ),
            None,
        )
        == "tenant-a"
    )


def test_get_current_tenant_skips_origin_check_for_safe_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        auth_jwt_secret=SECRET,
        auth_base_url="https://onramp.example.com",
    )
    monkeypatch.setattr("app.api.deps.get_settings", lambda: settings)

    assert get_current_tenant(_request(method="GET", cookie=_token()), None) == "tenant-a"


def test_get_current_tenant_normalizes_origin_case(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        auth_jwt_secret=SECRET,
        auth_base_url="HTTPS://OnRamp.Example.Com",
    )
    monkeypatch.setattr("app.api.deps.get_settings", lambda: settings)

    assert (
        get_current_tenant(
            _request(
                method="POST",
                cookie=_token(),
                origin="https://onramp.example.com",
            ),
            None,
        )
        == "tenant-a"
    )


def test_get_current_tenant_returns_503_when_origin_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(auth_jwt_secret=SECRET, auth_base_url="")
    monkeypatch.setattr("app.api.deps.get_settings", lambda: settings)

    with pytest.raises(HTTPException) as exc_info:
        get_current_tenant(
            _request(method="POST", cookie=_token(), origin="https://onramp.example.com"),
            None,
        )

    assert exc_info.value.status_code == 503


@pytest.mark.parametrize("origin", [None, "https://attacker.example.com"])
def test_get_current_tenant_rejects_untrusted_cookie_origin(
    monkeypatch: pytest.MonkeyPatch,
    origin: str | None,
) -> None:
    settings = Settings(
        auth_jwt_secret=SECRET,
        auth_base_url="https://onramp.example.com",
    )
    monkeypatch.setattr("app.api.deps.get_settings", lambda: settings)

    with pytest.raises(HTTPException) as exc_info:
        get_current_tenant(
            _request(method="POST", cookie=_token(), origin=origin),
            None,
        )

    assert exc_info.value.status_code == 403
