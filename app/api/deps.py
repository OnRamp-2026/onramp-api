from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import session_scope
from app.storage.base import ObjectStorage
from app.storage.factory import get_storage


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async for session in session_scope():
        yield session


def get_current_tenant(
    x_tenant_id: Annotated[
        str,
        Header(
            alias="X-Tenant-ID",
            min_length=1,
            max_length=128,
            pattern=r"^[0-9A-Za-z_-]+$",
        ),
    ],
) -> str:
    """현재 인증 계층의 tenant adapter.

    JWT 인증이 도입되면 이 함수만 claims 기반 구현으로 교체한다.
    """
    return x_tenant_id


def get_object_storage() -> ObjectStorage:
    return get_storage()


DatabaseSession = Annotated[AsyncSession, Depends(get_db_session)]
CurrentTenant = Annotated[str, Depends(get_current_tenant)]
StorageDependency = Annotated[ObjectStorage, Depends(get_object_storage)]
