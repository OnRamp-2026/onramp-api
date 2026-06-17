from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode

import httpx
import jwt

from app.auth.session_tokens import decode_auth_state, issue_auth_state, issue_session_token
from app.auth.tenant_registry import TenantContext, resolve_tenant_context
from app.config import Settings
from app.middleware.error_handler import OnRampError
from app.models.auth import AuthSessionResponse, SlackAuthorizeResponse

SLACK_PROVIDER = "slack"
SLACK_SCOPE = "openid profile email"
SLACK_TEAM_ID_CLAIM = "https://slack.com/team_id"
SLACK_USER_ID_CLAIM = "https://slack.com/user_id"


@dataclass(frozen=True)
class SlackTokenExchange:
    access_token: str
    id_token: str


def build_slack_authorization(
    settings: Settings,
    *,
    team: str | None = None,
    expected_host: str = "",
) -> SlackAuthorizeResponse:
    _validate_slack_config(settings, require_client_secret=False)
    state_token, state = issue_auth_state(
        provider=SLACK_PROVIDER,
        settings=settings,
        expected_host=expected_host,
    )
    params = {
        "response_type": "code",
        "client_id": settings.auth_slack_client_id,
        "scope": SLACK_SCOPE,
        "state": state_token,
        "nonce": state.nonce,
        "redirect_uri": settings.auth_slack_redirect_uri,
    }
    if team:
        params["team"] = team
    return SlackAuthorizeResponse(
        authorization_url=f"{settings.auth_slack_authorize_url}?{urlencode(params)}",
        state_expires_at=state.expires_at,
    )


async def authenticate_with_slack_callback(
    *,
    code: str,
    state: str,
    settings: Settings,
    callback_host: str = "",
) -> AuthSessionResponse:
    _validate_slack_config(settings, require_client_secret=True)
    auth_state = decode_auth_state(state, provider=SLACK_PROVIDER, settings=settings)
    token_exchange = await _exchange_code_for_token(code=code, settings=settings)
    claims = _decode_slack_id_token(
        token_exchange.id_token,
        expected_nonce=auth_state.nonce,
        settings=settings,
    )
    external_tenant = _required_claim(claims, SLACK_TEAM_ID_CLAIM, "Slack team_id")
    user_id = _required_claim(claims, SLACK_USER_ID_CLAIM, "Slack user_id")
    try:
        tenant_context = resolve_tenant_context(
            provider=SLACK_PROVIDER,
            external_tenant=external_tenant,
            settings=settings,
        )
    except ValueError as exc:
        raise OnRampError("등록되지 않은 Slack workspace입니다.", status_code=403) from exc
    _validate_tenant_host(
        tenant_context,
        expected_host=auth_state.expected_host,
        callback_host=callback_host,
    )
    tenant_id = tenant_context.tenant_id

    session = issue_session_token(
        tenant_id=tenant_id,
        provider=SLACK_PROVIDER,
        external_tenant=external_tenant,
        user_id=user_id,
        settings=settings,
    )
    return AuthSessionResponse(
        access_token=session.access_token,
        expires_at=session.expires_at,
        tenant_id=tenant_id,
        tenant_api_base_url=tenant_context.tenant_api_base_url,
        provider=SLACK_PROVIDER,
        external_tenant=external_tenant,
        user_id=user_id,
        email=_optional_string_claim(claims, "email"),
        email_verified=bool(claims.get("email_verified", False)),
        name=_optional_string_claim(claims, "name"),
    )


def _validate_tenant_host(tenant_context: TenantContext, *, expected_host: str, callback_host: str) -> None:
    if not tenant_context.allowed_hosts:
        return
    normalized_expected = _normalize_host(expected_host)
    normalized_callback = _normalize_host(callback_host)
    tenant_host = normalized_expected or normalized_callback
    if tenant_host and tenant_host not in tenant_context.allowed_hosts:
        raise OnRampError("로그인 요청 host가 tenant registry와 일치하지 않습니다.", status_code=403)
    if normalized_callback and normalized_expected and normalized_callback != normalized_expected:
        raise OnRampError("로그인 callback host가 state와 일치하지 않습니다.", status_code=403)


def _normalize_host(value: str) -> str:
    return value.strip().lower().rstrip(".")


async def _exchange_code_for_token(*, code: str, settings: Settings) -> SlackTokenExchange:
    payload = {
        "client_id": settings.auth_slack_client_id,
        "client_secret": settings.auth_slack_client_secret.get_secret_value(),
        "code": code,
        "redirect_uri": settings.auth_slack_redirect_uri,
        "grant_type": "authorization_code",
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Slack openid.connect.token은 application/x-www-form-urlencoded만 파싱한다(JSON 바디는 code 미인식 → invalid_code).
            response = await client.post(settings.auth_slack_token_url, data=payload)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OnRampError("Slack 토큰 교환 호출에 실패했습니다.", status_code=502) from exc
    data = response.json()
    if not data.get("ok"):
        error = str(data.get("error", "unknown_error"))
        status_code = 401 if error in {"invalid_code", "invalid_grant", "access_denied", "invalid_request"} else 502
        raise OnRampError(f"Slack 토큰 교환에 실패했습니다: {error}", status_code=status_code)
    access_token = data.get("access_token")
    id_token = data.get("id_token")
    if not isinstance(access_token, str) or not access_token:
        raise OnRampError("Slack 토큰 응답에 access_token이 없습니다.", status_code=502)
    if not isinstance(id_token, str) or not id_token:
        raise OnRampError("Slack 토큰 응답에 id_token이 없습니다.", status_code=502)
    return SlackTokenExchange(access_token=access_token, id_token=id_token)


def _decode_slack_id_token(id_token: str, *, expected_nonce: str, settings: Settings) -> dict[str, Any]:
    try:
        claims = jwt.decode(
            id_token,
            options={
                "verify_signature": False,
                "verify_exp": False,
                "verify_nbf": False,
                "verify_iat": False,
                "verify_aud": False,
                "verify_iss": False,
            },
            algorithms=["RS256", "HS256"],
        )
    except jwt.InvalidTokenError as exc:
        raise OnRampError("Slack id_token이 유효하지 않습니다.", status_code=401) from exc
    if claims.get("iss") != settings.auth_slack_issuer:
        raise OnRampError("Slack id_token issuer가 일치하지 않습니다.", status_code=401)
    if claims.get("aud") != settings.auth_slack_client_id:
        raise OnRampError("Slack id_token audience가 일치하지 않습니다.", status_code=401)
    if claims.get("nonce") != expected_nonce:
        raise OnRampError("Slack id_token nonce가 일치하지 않습니다.", status_code=401)
    exp = claims.get("exp")
    if not isinstance(exp, int) or exp <= int(datetime.now(UTC).timestamp()):
        raise OnRampError("Slack id_token이 만료되었습니다.", status_code=401)
    return claims


def _required_claim(claims: dict[str, Any], key: str, label: str) -> str:
    value = claims.get(key)
    if not isinstance(value, str) or not value:
        raise OnRampError(f"{label} claim이 없습니다.", status_code=401)
    return value


def _optional_string_claim(claims: dict[str, Any], key: str) -> str:
    value = claims.get(key)
    return value if isinstance(value, str) else ""


def _validate_slack_config(settings: Settings, *, require_client_secret: bool) -> None:
    if not settings.auth_slack_client_id or not settings.auth_slack_redirect_uri:
        raise OnRampError("Slack 로그인 설정이 구성되지 않았습니다.", status_code=503)
    if require_client_secret and not settings.auth_slack_client_secret.get_secret_value():
        raise OnRampError("Slack 로그인 설정이 구성되지 않았습니다.", status_code=503)
