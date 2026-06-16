from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.session import (
    SessionUser,
    decode_session_claims,
    extract_token,
    get_current_user,
)
from app.config import Settings, get_settings
from app.db.postgres import session_scope
from app.storage.base import ObjectStorage
from app.storage.factory import get_storage

bearer_scheme = HTTPBearer(auto_error=False)
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with session_scope() as session:
        yield session


def decode_tenant_token(token: str, settings: Settings) -> str:
    return str(decode_session_claims(token, settings)["tenant_id"])


def _validate_cookie_origin(request: Request, settings: Settings) -> None:
    if request.method.upper() in _SAFE_METHODS:
        return

    configured = urlsplit(settings.auth_base_url)
    expected_origin = (
        f"{configured.scheme.lower()}://{configured.netloc.lower()}" if configured.scheme and configured.netloc else ""
    )
    if not expected_origin:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="쿠키 인증 Origin 검증이 구성되지 않았습니다.",
        )
    if request.headers.get("origin", "").lower() != expected_origin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="허용되지 않은 요청 Origin입니다.",
        )


def get_current_tenant(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer_scheme)],
) -> str:
    settings = get_settings()
    cookie_token = request.cookies.get(settings.auth_cookie_name)
    if cookie_token:
        _validate_cookie_origin(request, settings)
        return decode_tenant_token(cookie_token, settings)

    if credentials is not None:
        return decode_tenant_token(credentials.credentials, settings)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="세션 쿠키 또는 Bearer 인증 토큰이 필요합니다.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_optional_user(request: Request) -> SessionUser | None:
    """로그인했으면 SessionUser, 아니면 None(401 미발생). 대화 기록 옵셔널 저장용 — 익명 챗은 그대로 동작."""
    settings = get_settings()
    token = extract_token(request, settings)
    if not token:
        return None
    try:
        claims = decode_session_claims(token, settings)
    except HTTPException:
        return None
    return SessionUser(
        tenant_id=claims["tenant_id"],
        subject=str(claims.get("sub", "")),
        provider=claims.get("provider"),
        name=claims.get("name"),
        email=claims.get("email"),
        claims=claims,
    )


def get_object_storage() -> ObjectStorage:
    return get_storage()


DatabaseSession = Annotated[AsyncSession, Depends(get_db_session)]
CurrentTenant = Annotated[str, Depends(get_current_tenant)]
CurrentUser = Annotated[SessionUser, Depends(get_current_user)]
OptionalUser = Annotated[SessionUser | None, Depends(get_optional_user)]
StorageDependency = Annotated[ObjectStorage, Depends(get_object_storage)]
