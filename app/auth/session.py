"""세션 JWT 발급·검증.

설계: 인증 서버를 만들지 않고 IdP(Slack)에 위임 — OnRamp는 IdP 검증 후 *우리 키로 서명한*
stateless 세션 JWT를 발급한다. 매 요청은 서명만 로컬 검증(EKS 수평 확장 자유, 세션 DB 없음).
검증 클레임 계약은 `app/api/v1` 의존성(`get_current_tenant`)과 동일: `exp` + `tenant_id` 필수.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from fastapi import HTTPException, Request, status
from jwt import InvalidTokenError

from app.config import Settings, get_settings

ALGORITHM = "HS256"
_TENANT_MAX_LEN = 128


@dataclass(frozen=True)
class SessionUser:
    tenant_id: str
    subject: str
    provider: str | None
    name: str | None
    email: str | None
    claims: dict[str, Any]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _require_secret(settings: Settings) -> str:
    secret = settings.auth_jwt_secret.get_secret_value()
    if len(secret) < 32:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="인증 설정이 구성되지 않았습니다.",
        )
    return secret


def _valid_tenant(tenant_id: Any) -> bool:
    return (
        isinstance(tenant_id, str)
        and 0 < len(tenant_id) <= _TENANT_MAX_LEN
        and all(ch.isalnum() or ch in "_-" for ch in tenant_id)
    )


def issue_session_token(
    *,
    tenant_id: str,
    subject: str,
    settings: Settings,
    provider: str | None = None,
    name: str | None = None,
    email: str | None = None,
    ttl_seconds: int | None = None,
) -> tuple[str, int]:
    """세션 JWT 발급. 반환: (토큰, 만료까지 초)."""
    secret = _require_secret(settings)
    if not _valid_tenant(tenant_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="tenant_id 형식이 유효하지 않습니다.")

    ttl = ttl_seconds if ttl_seconds is not None else settings.auth_session_ttl_seconds
    now = _utcnow()
    claims: dict[str, Any] = {
        "sub": subject,
        "tenant_id": tenant_id,
        "iat": now,
        "exp": now + timedelta(seconds=ttl),
    }
    if settings.auth_jwt_issuer:
        claims["iss"] = settings.auth_jwt_issuer
    if settings.auth_jwt_audience:
        claims["aud"] = settings.auth_jwt_audience
    if provider:
        claims["provider"] = provider
    if name:
        claims["name"] = name
    if email:
        claims["email"] = email

    return jwt.encode(claims, secret, algorithm=ALGORITHM), ttl


def decode_session_claims(token: str, settings: Settings) -> dict[str, Any]:
    secret = _require_secret(settings)
    try:
        claims: dict[str, Any] = jwt.decode(
            token,
            secret,
            algorithms=[ALGORITHM],
            audience=settings.auth_jwt_audience or None,
            issuer=settings.auth_jwt_issuer or None,
            options={
                "require": ["exp", "tenant_id"],
                "verify_aud": bool(settings.auth_jwt_audience),
            },
        )
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 인증 토큰입니다.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    if not _valid_tenant(claims.get("tenant_id")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="tenant claim이 유효하지 않습니다.")
    return claims


def extract_token(request: Request, settings: Settings) -> str | None:
    """세션 토큰을 httpOnly 쿠키 우선, 없으면 Authorization Bearer에서 추출."""
    cookie = request.cookies.get(settings.auth_cookie_name)
    if cookie:
        return cookie
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer" and value:
        return value
    return None


def get_current_user(request: Request) -> SessionUser:
    settings = get_settings()
    token = extract_token(request, settings)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="인증이 필요합니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    claims = decode_session_claims(token, settings)
    return SessionUser(
        tenant_id=claims["tenant_id"],
        subject=str(claims.get("sub", "")),
        provider=claims.get("provider"),
        name=claims.get("name"),
        email=claims.get("email"),
        claims=claims,
    )
