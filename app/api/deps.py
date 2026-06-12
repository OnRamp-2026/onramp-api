from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.postgres import session_scope
from app.storage.base import ObjectStorage
from app.storage.factory import get_storage

bearer_scheme = HTTPBearer(auto_error=False)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async for session in session_scope():
        yield session


def decode_tenant_token(token: str, settings: Settings) -> str:
    secret = settings.auth_jwt_secret.get_secret_value()
    if len(secret) < 32:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="인증 설정이 구성되지 않았습니다.",
        )

    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
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

    tenant_id = claims.get("tenant_id")
    if not isinstance(tenant_id, str) or not tenant_id or len(tenant_id) > 128:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="tenant claim이 유효하지 않습니다.")
    if not all(character.isalnum() or character in "_-" for character in tenant_id):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="tenant claim이 유효하지 않습니다.")
    return tenant_id


def get_current_tenant(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer_scheme)],
) -> str:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer 인증 토큰이 필요합니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_tenant_token(credentials.credentials, get_settings())


def get_object_storage() -> ObjectStorage:
    return get_storage()


DatabaseSession = Annotated[AsyncSession, Depends(get_db_session)]
CurrentTenant = Annotated[str, Depends(get_current_tenant)]
StorageDependency = Annotated[ObjectStorage, Depends(get_object_storage)]
