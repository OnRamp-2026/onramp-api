from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.api.deps import decode_tenant_token, get_current_tenant
from app.config import Settings

SECRET = "test-auth-secret-with-at-least-32-bytes"


def _token(**claims: object) -> str:
    payload = {
        "tenant_id": "tenant-a",
        "aud": "onramp-api",
        "exp": datetime.now(UTC) + timedelta(minutes=5),
        **claims,
    }
    return jwt.encode(payload, SECRET, algorithm="HS256")


def test_decode_tenant_token_uses_verified_claim() -> None:
    settings = Settings(_env_file=None, debug=False, auth_jwt_secret=SECRET)

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
    settings = Settings(_env_file=None, debug=False, auth_jwt_secret=SECRET)

    with pytest.raises(HTTPException) as exc_info:
        decode_tenant_token(token, settings)

    assert exc_info.value.status_code == 401


def test_decode_tenant_token_requires_auth_configuration() -> None:
    with pytest.raises(HTTPException) as exc_info:
        decode_tenant_token(_token(), Settings(_env_file=None, debug=False, auth_jwt_secret=""))

    assert exc_info.value.status_code == 503


def test_get_current_tenant_requires_bearer_credentials() -> None:
    with pytest.raises(HTTPException) as exc_info:
        get_current_tenant(None)

    assert exc_info.value.status_code == 401


def test_get_current_tenant_does_not_accept_client_tenant_header(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None, debug=False, auth_jwt_secret=SECRET)
    monkeypatch.setattr("app.api.deps.get_settings", lambda: settings)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=_token())

    assert get_current_tenant(credentials) == "tenant-a"
