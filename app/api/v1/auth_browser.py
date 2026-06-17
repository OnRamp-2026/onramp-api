"""브라우저 SPA 로그인용 쿠키 세션 플로우.

shared-auth(#190)는 API/토큰(Bearer/JSON) 클라이언트용 — `/v1/auth/slack/*`, `/auth/callback`(JSON).
브라우저 SPA는 httpOnly 쿠키 세션 + 화면 복귀(redirect)가 필요하므로, 이 라우터가 그 계약을 제공한다.

내부적으로 shared-auth 빌딩블록(`build_slack_authorization`/`authenticate_with_slack_callback`)을 **그대로 호출**하고,
쿠키에 심을 세션 토큰만 기존 `app.auth.session.issue_session_token`(name/email 포함)으로 재발급한다 —
deps(`get_current_user`/`get_current_tenant`)가 그 토큰을 검증하므로 chat/conversations 인증과 호환된다.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from app.auth.session import get_current_user, issue_session_token
from app.auth.slack_oidc import authenticate_with_slack_callback, build_slack_authorization
from app.config import Settings, get_settings
from app.middleware.error_handler import OnRampError

browser_router = APIRouter(prefix="/auth", tags=["Auth"])


class MeResponse(BaseModel):
    tenant_id: str
    subject: str
    provider: str | None = None
    name: str | None = None
    email: str | None = None


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


@browser_router.get("/login")
async def login(team: str | None = Query(default=None, max_length=128)) -> RedirectResponse:
    """Slack authorize URL로 브라우저를 보낸다(307). 프론트 진입점."""
    authorization = build_slack_authorization(get_settings(), team=team)
    return RedirectResponse(authorization.authorization_url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@browser_router.get("/callback")
async def callback(
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
) -> RedirectResponse:
    """Slack redirect 대상. 세션 쿠키 set 후 프론트로 복귀(303). (JSON 토큰은 /v1/auth/slack/callback)"""
    settings = get_settings()
    if error:
        raise OnRampError(f"Slack 로그인에 실패했습니다: {error}", status_code=401)
    if not code or not state:
        raise OnRampError("Slack callback에 code/state가 없습니다.", status_code=400)
    session = await authenticate_with_slack_callback(code=code, state=state, settings=settings)
    # 쿠키 토큰은 deps 호환 issuer로 재발급(name/email 포함 → /me·사이드바 표시)
    token, ttl = issue_session_token(
        tenant_id=session.tenant_id,
        subject=session.user_id,
        settings=settings,
        provider=session.provider,
        name=session.name or None,
        email=session.email or None,
    )
    response = RedirectResponse(settings.frontend_post_login_redirect, status_code=status.HTTP_303_SEE_OTHER)
    _set_session_cookie(response, token, ttl, settings)
    return response


@browser_router.get("/me", response_model=MeResponse)
async def me(request: Request) -> MeResponse:
    """세션 쿠키(또는 Bearer) 기준 현재 사용자."""
    user = get_current_user(request)
    return MeResponse(
        tenant_id=user.tenant_id,
        subject=user.subject,
        provider=user.provider,
        name=user.name,
        email=user.email,
    )


@browser_router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response) -> Response:
    """세션 쿠키 삭제(204). stateless라 서버 세션 없음."""
    response.delete_cookie(key=get_settings().auth_cookie_name, path="/")
    response.status_code = status.HTTP_204_NO_CONTENT
    return response
