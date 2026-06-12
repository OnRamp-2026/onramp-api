from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_current_tenant, get_db_session, get_object_storage
from app.api.v1.transcriptions import router as transcriptions_router
from app.db.base import Base
from app.middleware.error_handler import register_error_handlers
from app.storage.base import ObjectMetadata, PresignedUpload

app = FastAPI()
register_error_handlers(app)
app.include_router(transcriptions_router, prefix="/v1")


class FakeObjectStorage:
    def __init__(self) -> None:
        self.objects: dict[str, ObjectMetadata] = {}

    async def create_presigned_upload(
        self,
        object_key: str,
        *,
        content_type: str,
        expires_in_seconds: int,
    ) -> PresignedUpload:
        return PresignedUpload(
            method="PUT",
            url=f"https://storage.test/{object_key}",
            headers={"Content-Type": content_type},
            expires_at=datetime.now(UTC) + timedelta(seconds=expires_in_seconds),
        )

    async def head(self, object_key: str) -> ObjectMetadata:
        return self.objects[object_key]


@pytest.fixture
async def transcription_client() -> AsyncIterator[tuple[AsyncClient, FakeObjectStorage]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    storage = FakeObjectStorage()

    async def override_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_session
    app.dependency_overrides[get_object_storage] = lambda: storage
    app.dependency_overrides[get_current_tenant] = lambda: "tenant-a"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, storage

    app.dependency_overrides.clear()
    await engine.dispose()


def request_body() -> dict[str, object]:
    return {
        "filename": "meeting.m4a",
        "content_type": "audio/mp4",
        "size_bytes": 1024,
        "title": "장애 대응 회의",
        "language": "ko-KR",
        "category": "장애대응",
    }


@pytest.mark.asyncio
async def test_create_transcription_returns_201_then_200_for_same_idempotency_key(
    transcription_client: tuple[AsyncClient, FakeObjectStorage],
) -> None:
    client, _ = transcription_client

    first = await client.post(
        "/v1/transcriptions",
        headers={"Idempotency-Key": "request-1"},
        json=request_body(),
    )
    second = await client.post(
        "/v1/transcriptions",
        headers={"Idempotency-Key": "request-1"},
        json=request_body(),
    )

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["workflow_id"] == first.json()["workflow_id"]
    assert second.json()["transcription_id"] == first.json()["transcription_id"]
    assert first.json()["upload"]["method"] == "PUT"
    expected_prefix = f"tenants/tenant-a/transcriptions/{first.json()['transcription_id']}/source/"
    assert expected_prefix in first.json()["upload"]["url"]


@pytest.mark.asyncio
async def test_upload_complete_and_status_lookup(
    transcription_client: tuple[AsyncClient, FakeObjectStorage],
) -> None:
    client, storage = transcription_client
    created = (await client.post("/v1/transcriptions", json=request_body())).json()
    transcription_id = created["transcription_id"]
    object_key = created["upload"]["url"].removeprefix("https://storage.test/")
    storage.objects[object_key] = ObjectMetadata(
        object_key=object_key,
        size_bytes=1024,
        content_type="audio/mp4",
        etag='"etag-1"',
    )

    completed = await client.post(
        f"/v1/transcriptions/{transcription_id}/upload-complete",
        json={"etag": '"etag-1"', "size_bytes": 1024},
    )
    status = await client.get(f"/v1/transcriptions/{transcription_id}")

    assert completed.status_code == 202
    assert completed.json()["status"] == "queued"
    assert status.status_code == 200
    assert status.json()["status"] == "queued"


@pytest.mark.asyncio
async def test_queued_idempotent_request_does_not_reissue_upload_url(
    transcription_client: tuple[AsyncClient, FakeObjectStorage],
) -> None:
    client, storage = transcription_client
    headers = {"Idempotency-Key": "request-queued"}
    created = (await client.post("/v1/transcriptions", headers=headers, json=request_body())).json()
    object_key = created["upload"]["url"].removeprefix("https://storage.test/")
    storage.objects[object_key] = ObjectMetadata(
        object_key=object_key,
        size_bytes=1024,
        content_type="audio/mp4",
        etag='"etag-1"',
    )
    await client.post(
        f"/v1/transcriptions/{created['transcription_id']}/upload-complete",
        json={"etag": '"etag-1"', "size_bytes": 1024},
    )

    repeated = await client.post("/v1/transcriptions", headers=headers, json=request_body())

    assert repeated.status_code == 200
    assert repeated.json()["status"] == "queued"
    assert repeated.json()["upload"] is None


@pytest.mark.asyncio
async def test_tenant_cannot_access_another_tenants_workflow(
    transcription_client: tuple[AsyncClient, FakeObjectStorage],
) -> None:
    client, _ = transcription_client
    created = (await client.post("/v1/transcriptions", json=request_body())).json()
    app.dependency_overrides[get_current_tenant] = lambda: "tenant-b"

    response = await client.get(f"/v1/transcriptions/{created['transcription_id']}")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_tenant_cannot_complete_another_tenants_upload(
    transcription_client: tuple[AsyncClient, FakeObjectStorage],
) -> None:
    client, _ = transcription_client
    created = (await client.post("/v1/transcriptions", json=request_body())).json()
    app.dependency_overrides[get_current_tenant] = lambda: "tenant-b"

    response = await client.post(
        f"/v1/transcriptions/{created['transcription_id']}/upload-complete",
        json={"etag": '"etag-1"', "size_bytes": 1024},
    )

    assert response.status_code == 404
