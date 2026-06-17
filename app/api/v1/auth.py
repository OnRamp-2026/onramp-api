from __future__ import annotations

from fastapi import APIRouter, Query, Request

from app.auth.slack_oidc import authenticate_with_slack_callback, build_slack_authorization
from app.config import get_settings
from app.middleware.error_handler import OnRampError
from app.models.auth import AuthSessionResponse, SlackAuthorizeResponse

router = APIRouter(prefix="/auth", tags=["Auth"])
public_router = APIRouter(prefix="/auth", tags=["Auth"])


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


@public_router.get("/callback", response_model=AuthSessionResponse)
async def public_callback(
    request: Request,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
) -> AuthSessionResponse:
    return await _slack_callback(request=request, code=code, state=state, error=error)


def _request_host(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-host")
    raw = forwarded.split(",", 1)[0].strip() if forwarded else request.headers.get("host", "")
    return raw.lower().rstrip(".")
