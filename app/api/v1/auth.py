from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field

from app.auth.session import issue_session_token
from app.auth.slack_oidc import authenticate_with_slack_callback, build_slack_authorization
from app.config import get_settings
from app.middleware.error_handler import OnRampError
from app.models.auth import AuthSessionResponse, SlackAuthorizeResponse

router = APIRouter(prefix="/auth", tags=["Auth"])
public_router = APIRouter(prefix="/auth", tags=["Auth"])


class DevTokenRequest(BaseModel):
    tenant_id: str | None = Field(default=None, description="미지정 시 auth_default_tenant")
    subject: str = Field(default="dev-user")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


@router.get("/slack/authorize", response_model=SlackAuthorizeResponse)
async def slack_authorize(
    request: Request,
    team: str | None = Query(default=None, max_length=128, description="선택적 Slack workspace(team) 힌트"),
) -> SlackAuthorizeResponse:
    return build_slack_authorization(get_settings(), team=team, expected_host=_request_host(request))


async def _slack_callback(
    request: Request,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
) -> AuthSessionResponse:
    if error:
        raise OnRampError(f"Slack 로그인에 실패했습니다: {error}", status_code=401)
    if not code or not state:
        raise OnRampError("Slack callback에 code/state가 없습니다.", status_code=400)
    return await authenticate_with_slack_callback(
        code=code,
        state=state,
        settings=get_settings(),
        callback_host=_request_host(request),
    )


@router.get("/slack/callback", response_model=AuthSessionResponse)
async def slack_callback(
    request: Request,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
) -> AuthSessionResponse:
    return await _slack_callback(request=request, code=code, state=state, error=error)


@router.post("/dev-token", response_model=TokenResponse)
async def dev_token(body: DevTokenRequest, response: Response) -> TokenResponse:
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
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=token,
        max_age=ttl,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite=settings.auth_cookie_samesite,
        path="/",
    )
    return TokenResponse(access_token=token, expires_in=ttl)


@public_router.get("/callback", response_model=AuthSessionResponse)
async def public_callback(
    request: Request,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
) -> AuthSessionResponse:
    return await _slack_callback(request=request, code=code, state=state, error=error)


def _request_host(request: Request) -> str:
    return request.headers.get("host", "").strip().lower().rstrip(".")
