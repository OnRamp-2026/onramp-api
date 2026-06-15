"""`/auth/*` — OIDC RP(Slack "Sign in with Slack") + 세션 발급.

인증 서버를 만들지 않는다. Slack(IdP)에 위임하고 onramp-api는 클라이언트(RP)로서:
  login → Slack authorize redirect / callback → id_token 검증 → 우리 세션 JWT 발급(httpOnly 쿠키)
프론트(onramp-web)는 `/auth/login` 이동·`/auth/me`·`/auth/logout`을 credentials:include(쿠키)로 호출.
세션 JWT 검증 계약은 `app/auth/session.py` 참조.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import jwt
from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import RedirectResponse
from jwt import InvalidTokenError
from pydantic import BaseModel, Field

from app.auth.session import (
    ALGORITHM,
    SessionUser,
    get_current_user,
    issue_session_token,
)
from app.config import Settings, get_settings

router = APIRouter()

# Slack OpenID Connect 엔드포인트
SLACK_AUTHORIZE = "https://slack.com/openid/connect/authorize"
SLACK_TOKEN = "https://slack.com/api/openid.connect.token"
SLACK_JWKS = "https://slack.com/openid/connect/keys"
SLACK_ISSUER = "https://slack.com"
SLACK_TEAM_CLAIM = "https://slack.com/team_id"
SLACK_SCOPE = "openid profile email"
_STATE_TTL = 600  # CSRF state 10분


# ── 모델 ──────────────────────────────────────────────────────────
class DevTokenRequest(BaseModel):
    tenant_id: str | None = Field(default=None, description="미지정 시 auth_default_tenant")
    subject: str = Field(default="dev-user")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class MeResponse(BaseModel):
    tenant_id: str
    subject: str
    provider: str | None = None
    name: str | None = None
    email: str | None = None


# ── 헬퍼 ──────────────────────────────────────────────────────────
def _set_session_cookie(response: Response, token: str, ttl: int, settings: Settings) -> None:
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=token,
        max_age=ttl,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite=settings.auth_cookie_samesite,
        path="/",
    )


def _safe_redirect_path(value: str | None) -> str:
    # open-redirect 방지: 사이트 내부 경로만 허용
    if value and value.startswith("/") and not value.startswith("//"):
        return value
    return "/"


def _sign_state(redirect: str, settings: Settings) -> str:
    secret = settings.auth_jwt_secret.get_secret_value()
    now = datetime.now(UTC)
    return jwt.encode(
        {"redirect": redirect, "iat": now, "exp": now + timedelta(seconds=_STATE_TTL)},
        secret,
        algorithm=ALGORITHM,
    )


def _verify_state(state: str, settings: Settings) -> str:
    try:
        claims = jwt.decode(state, settings.auth_jwt_secret.get_secret_value(), algorithms=[ALGORITHM])
    except InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="state 검증 실패(CSRF).") from exc
    return _safe_redirect_path(claims.get("redirect"))


def _require_slack(settings: Settings) -> str:
    if not settings.slack_client_id or not settings.slack_client_secret.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Slack OIDC가 구성되지 않았습니다. (slack_client_id/secret·auth_base_url 필요)",
        )
    if not settings.auth_base_url:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="auth_base_url 미설정.")
    return f"{settings.auth_base_url.rstrip('/')}/auth/callback"


# ── 엔드포인트 ────────────────────────────────────────────────────
@router.get("/login")
async def login(
    provider: str = Query(default="slack"),
    redirect: str | None = Query(default=None),
) -> RedirectResponse:
    """Slack authorize로 리다이렉트(브라우저 top-level)."""
    settings = get_settings()
    if provider != "slack":
        # 회사 SSO(Keycloak/Entra)는 동일 RP 패턴으로 확장 예정(P1)
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=f"provider '{provider}' 미지원.")
    redirect_uri = _require_slack(settings)
    state = _sign_state(_safe_redirect_path(redirect), settings)
    params = {
        "response_type": "code",
        "client_id": settings.slack_client_id,
        "scope": SLACK_SCOPE,
        "redirect_uri": redirect_uri,
        "state": state,
    }
    return RedirectResponse(url=str(httpx.URL(SLACK_AUTHORIZE, params=params)))


@router.get("/callback")
async def callback(request: Request, code: str = Query(...), state: str = Query(...)) -> RedirectResponse:
    """Slack code → id_token 검증 → 세션 JWT 발급(쿠키) → 프론트 복귀."""
    settings = get_settings()
    redirect_uri = _require_slack(settings)
    post_login = _verify_state(state, settings)

    async with httpx.AsyncClient(timeout=10) as client:
        token_res = await client.post(
            SLACK_TOKEN,
            data={
                "client_id": settings.slack_client_id,
                "client_secret": settings.slack_client_secret.get_secret_value(),
                "code": code,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
    payload = token_res.json()
    if not payload.get("ok", False) or "id_token" not in payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Slack 토큰 교환 실패.")

    claims = _verify_slack_id_token(payload["id_token"], settings)
    team_id = claims.get(SLACK_TEAM_CLAIM)
    if not isinstance(team_id, str) or not team_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="team_id(테넌트) 클레임 없음.")

    # 인가 L1: 워크스페이스(team_id) = 회사 = 테넌트
    token, ttl = issue_session_token(
        tenant_id=team_id,
        subject=str(claims.get("sub", "")),
        settings=settings,
        provider="slack",
        name=claims.get("name"),
        email=claims.get("email"),
    )
    response = RedirectResponse(url=f"{settings.auth_base_url.rstrip('/')}{post_login}")
    _set_session_cookie(response, token, ttl, settings)
    return response


def _verify_slack_id_token(id_token: str, settings: Settings) -> dict[str, Any]:
    try:
        signing_key = jwt.PyJWKClient(SLACK_JWKS).get_signing_key_from_jwt(id_token)
        return jwt.decode(
            id_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=settings.slack_client_id,
            issuer=SLACK_ISSUER,
        )
    except InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Slack id_token 검증 실패.") from exc


@router.get("/me", response_model=MeResponse)
async def me(request: Request) -> MeResponse:
    """세션 쿠키(또는 Bearer) 기준 현재 사용자."""
    user: SessionUser = get_current_user(request)
    return MeResponse(
        tenant_id=user.tenant_id,
        subject=user.subject,
        provider=user.provider,
        name=user.name,
        email=user.email,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response) -> Response:
    """세션 쿠키 삭제. (stateless라 서버 세션 없음 — 즉시 무효화는 P1 Redis denylist)"""
    settings = get_settings()
    response.delete_cookie(key=settings.auth_cookie_name, path="/")
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


@router.post("/dev-token", response_model=TokenResponse)
async def dev_token(body: DevTokenRequest, response: Response) -> TokenResponse:
    """Slack 없이 세션 발급(개발·STT 테스트용). auth_dev_token_enabled=true 일 때만."""
    settings = get_settings()
    if not settings.auth_dev_token_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    tenant_id = body.tenant_id or settings.auth_default_tenant
    token, ttl = issue_session_token(
        tenant_id=tenant_id,
        subject=body.subject,
        settings=settings,
        provider="dev",
    )
    _set_session_cookie(response, token, ttl, settings)  # 브라우저 dev 플로우용
    return TokenResponse(access_token=token, expires_in=ttl)  # Bearer 클라이언트(STT)용
