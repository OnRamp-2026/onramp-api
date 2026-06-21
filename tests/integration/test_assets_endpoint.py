from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_db_session
from app.api.v1.assets import router
from app.auth.session import SessionUser, get_current_user
from app.db.base import Base
from app.db.models import Report, ReportStatus, TranscriptionWorkflow, WorkflowStatus
from app.middleware.error_handler import register_error_handlers

app = FastAPI()
register_error_handlers(app)
app.include_router(router, prefix="/v1")


def user(user_id: str) -> SessionUser:
    return SessionUser(
        tenant_id="tenant-a",
        subject=user_id,
        provider="test",
        name="Test",
        email="test@example.com",
        claims={},
    )


def workflow(user_id: str, title: str) -> TranscriptionWorkflow:
    return TranscriptionWorkflow(
        transcription_id=uuid4(),
        tenant_id="tenant-a",
        created_by_user_id=user_id,
        idempotency_key=None,
        status=WorkflowStatus.transcribing,
        source_object_key=f"{title}.m4a",
        source_filename=f"{title}.m4a",
        source_content_type="audio/mp4",
        source_size_bytes=1024,
        title=title,
        language="ko-KR",
        category="장애대응",
    )


@pytest.fixture
async def assets_client() -> AsyncIterator[AsyncClient]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        session.add_all([workflow("user-a", "내 기록"), workflow("user-b", "다른 기록")])
        await session.commit()

    async def override_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_session
    app.dependency_overrides[get_current_user] = lambda: user("user-a")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.asyncio
async def test_assets_endpoint_returns_only_authenticated_users_history(assets_client: AsyncClient) -> None:
    response = await assets_client.get("/v1/assets")

    assert response.status_code == 200
    assert [item["title"] for item in response.json()["items"]] == ["내 기록"]
    assert response.json()["counts"]["all"] == 1


@pytest.mark.asyncio
async def test_assets_endpoint_rejects_empty_subject(assets_client: AsyncClient) -> None:
    app.dependency_overrides[get_current_user] = lambda: user("")

    response = await assets_client.get("/v1/assets")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_delete_draft_asset_returns_deleting() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    item = workflow("user-a", "삭제할 초안")
    item.status = WorkflowStatus.draft
    async with session_factory() as session:
        session.add(item)
        session.add(
            Report(
                tenant_id=item.tenant_id,
                source_transcription_id=item.transcription_id,
                title=item.title,
                category=item.category,
                situation="상황",
                cause="원인",
                evidence="근거",
                solution="해결",
                infra_context="환경",
                status=ReportStatus.draft,
                raw_text_sha256="a" * 64,
                corrected_text_sha256="b" * 64,
                dictionary_version="v1",
                result_object_key="result.json",
            )
        )
        await session.commit()

    async def override_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_session
    app.dependency_overrides[get_current_user] = lambda: user("user-a")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.delete(f"/v1/assets/{item.transcription_id}")
    app.dependency_overrides.clear()
    await engine.dispose()

    assert response.status_code == 202
    assert response.json() == {
        "transcription_id": str(item.transcription_id),
        "status": "deleting",
    }
