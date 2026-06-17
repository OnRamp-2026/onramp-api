from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from jwt import InvalidTokenError

from app.config import Settings
from app.middleware.error_handler import OnRampError

STATE_PURPOSE = "slack-auth-state"


@dataclass(frozen=True)
class AuthState:
    provider: str
    nonce: str
    expires_at: datetime
    expected_host: str = ""


@dataclass(frozen=True)
class IssuedSessionToken:
    access_token: str
    expires_at: datetime


def issue_auth_state(*, provider: str, settings: Settings, expected_host: str = "") -> tuple[str, AuthState]:
    secret = _auth_secret(settings)
    now = datetime.now(UTC)
    expires_at = now + timedelta(seconds=settings.auth_state_ttl_seconds)
    nonce = secrets.token_urlsafe(24)
    normalized_host = expected_host.strip().lower().rstrip(".")
    token = jwt.encode(
        {
            "purpose": STATE_PURPOSE,
            "provider": provider,
            "nonce": nonce,
            "expected_host": normalized_host,
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
        },
        secret,
        algorithm="HS256",
    )
    return token, AuthState(provider=provider, nonce=nonce, expires_at=expires_at, expected_host=normalized_host)


def decode_auth_state(state_token: str, *, provider: str, settings: Settings) -> AuthState:
    secret = _auth_secret(settings)
    try:
        claims = jwt.decode(
            state_token,
            secret,
            algorithms=["HS256"],
            options={"require": ["purpose", "provider", "nonce", "exp"]},
        )
    except InvalidTokenError as exc:
        raise OnRampError("유효하지 않거나 만료된 로그인 state입니다.", status_code=401) from exc
    if claims.get("purpose") != STATE_PURPOSE or claims.get("provider") != provider:
        raise OnRampError("유효하지 않은 로그인 state입니다.", status_code=401)
    nonce = claims.get("nonce")
    if not isinstance(nonce, str) or not nonce:
        raise OnRampError("유효하지 않은 로그인 state입니다.", status_code=401)
    exp = claims.get("exp")
    if not isinstance(exp, int):
        raise OnRampError("유효하지 않은 로그인 state입니다.", status_code=401)
    expected_host = claims.get("expected_host")
    return AuthState(
        provider=provider,
        nonce=nonce,
        expires_at=datetime.fromtimestamp(exp, UTC),
        expected_host=expected_host if isinstance(expected_host, str) else "",
    )


def issue_session_token(
    *,
    tenant_id: str,
    provider: str,
    external_tenant: str,
    user_id: str,
    settings: Settings,
) -> IssuedSessionToken:
    secret = _auth_secret(settings)
    now = datetime.now(UTC)
    expires_at = now + timedelta(seconds=settings.auth_session_ttl_seconds)
    claims: dict[str, Any] = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "provider": provider,
        "external_tenant": external_tenant,
        "iat": now,
        "exp": expires_at,
    }
    if settings.auth_jwt_audience:
        claims["aud"] = settings.auth_jwt_audience
    if settings.auth_jwt_issuer:
        claims["iss"] = settings.auth_jwt_issuer
    return IssuedSessionToken(
        access_token=jwt.encode(claims, secret, algorithm="HS256"),
        expires_at=expires_at,
    )


def _auth_secret(settings: Settings) -> str:
    secret = settings.auth_jwt_secret.get_secret_value()
    if len(secret) < 32:
        raise OnRampError("인증 설정이 구성되지 않았습니다.", status_code=503)
    return secret
