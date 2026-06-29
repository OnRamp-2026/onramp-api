from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_db_session, get_stt_client
from app.api.v1.transcriptions import router as transcriptions_router
from app.auth.session import SessionUser, get_current_user
from app.db.base import Base
from app.middleware.error_handler import register_error_handlers
from app.services.stt_result_client import (
    SttCompleteUploadResponse,
    SttCreateTranscriptionResponse,
    SttUploadInstruction,
)

app = FastAPI()
register_error_handlers(app)
app.include_router(transcriptions_router, prefix="/v1")


def current_user(*, tenant_id: str = "tenant-a", user_id: str = "user-a") -> SessionUser:
    return SessionUser(
        tenant_id=tenant_id,
        subject=user_id,
        provider="test",
        name="Test User",
        email="test@example.com",
        claims={},
    )


class FakeSttResultClient:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, object]] = []
        self.complete_calls: list[dict[str, object]] = []
        self.fail_complete = False

    async def create_transcription(
        self,
        *,
        tenant_id: str,
        transcription_id: UUID,
        filename: str,
        content_type: str,
        size_bytes: int,
        idempotency_key: str | None = None,
    ) -> SttCreateTranscriptionResponse:
        self.create_calls.append(
            {
                "tenant_id": tenant_id,
                "transcription_id": transcription_id,
                "filename": filename,
                "content_type": content_type,
                "size_bytes": size_bytes,
                "idempotency_key": idempotency_key,
            }
        )
        object_key = f"tenants/{tenant_id}/transcriptions/{transcription_id}/source/{filename}"
        return SttCreateTranscriptionResponse(
            transcription_id=transcription_id,
            status="awaiting_upload",
            source_object_key=object_key,
            upload=SttUploadInstruction(
                method="PUT",
                url=f"https://storage.test/{object_key}",
                headers={"Content-Type": content_type},
                expires_at=datetime.now(UTC) + timedelta(seconds=900),
            ),
        )

    async def complete_upload(
        self,
        transcription_id: UUID,
        *,
        tenant_id: str,
        etag: str | None,
        size_bytes: int,
    ) -> SttCompleteUploadResponse:
        self.complete_calls.append(
            {
                "tenant_id": tenant_id,
                "transcription_id": transcription_id,
                "etag": etag,
                "size_bytes": size_bytes,
            }
        )
        if self.fail_complete:
            raise httpx.HTTPStatusError(
                "409 Conflict",
                request=httpx.Request("POST", "http://stt"),
                response=httpx.Response(409),
            )
        return SttCompleteUploadResponse(
            transcription_id=transcription_id,
            status="queued",
        )


@pytest.fixture
async def transcription_client() -> AsyncIterator[tuple[AsyncClient, FakeSttResultClient]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    stt_client = FakeSttResultClient()

    async def override_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_session
    app.dependency_overrides[get_stt_client] = lambda: stt_client
    app.dependency_overrides[get_current_user] = current_user

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, stt_client

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
    transcription_client: tuple[AsyncClient, FakeSttResultClient],
) -> None:
    client, stt_client = transcription_client

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
    assert stt_client.create_calls[0]["tenant_id"] == "tenant-a"


@pytest.mark.asyncio
async def test_create_transcription_rejects_blank_idempotency_key(
    transcription_client: tuple[AsyncClient, FakeSttResultClient],
) -> None:
    client, _ = transcription_client

    response = await client.post(
        "/v1/transcriptions",
        headers={"Idempotency-Key": "   "},
        json=request_body(),
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_complete_and_status_lookup(
    transcription_client: tuple[AsyncClient, FakeSttResultClient],
) -> None:
    client, stt_client = transcription_client
    created = (await client.post("/v1/transcriptions", json=request_body())).json()
    transcription_id = created["transcription_id"]

    completed = await client.post(
        f"/v1/transcriptions/{transcription_id}/upload-complete",
        json={"etag": '"etag-1"', "size_bytes": 1024},
    )
    status = await client.get(f"/v1/transcriptions/{transcription_id}")

    assert completed.status_code == 202
    assert completed.json()["status"] == "queued"
    assert status.status_code == 200
    assert status.json()["status"] == "queued"
    assert stt_client.complete_calls == [
        {
            "tenant_id": "tenant-a",
            "transcription_id": UUID(transcription_id),
            "etag": '"etag-1"',
            "size_bytes": 1024,
        }
    ]


@pytest.mark.asyncio
async def test_upload_complete_rejects_metadata_mismatch_without_state_transition(
    transcription_client: tuple[AsyncClient, FakeSttResultClient],
) -> None:
    client, stt_client = transcription_client
    created = (await client.post("/v1/transcriptions", json=request_body())).json()
    transcription_id = created["transcription_id"]
    stt_client.fail_complete = True

    completed = await client.post(
        f"/v1/transcriptions/{transcription_id}/upload-complete",
        json={"etag": '"wrong-etag"', "size_bytes": 1024},
    )
    status = await client.get(f"/v1/transcriptions/{transcription_id}")

    assert completed.status_code == 409
    assert status.status_code == 200
    assert status.json()["status"] == "awaiting_upload"


@pytest.mark.asyncio
async def test_queued_idempotent_request_does_not_reissue_upload_url(
    transcription_client: tuple[AsyncClient, FakeSttResultClient],
) -> None:
    client, _ = transcription_client
    headers = {"Idempotency-Key": "request-queued"}
    created = (await client.post("/v1/transcriptions", headers=headers, json=request_body())).json()
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
    transcription_client: tuple[AsyncClient, FakeSttResultClient],
) -> None:
    client, _ = transcription_client
    created = (await client.post("/v1/transcriptions", json=request_body())).json()
    app.dependency_overrides[get_current_user] = lambda: current_user(tenant_id="tenant-b")

    response = await client.get(f"/v1/transcriptions/{created['transcription_id']}")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_tenant_cannot_complete_another_tenants_upload(
    transcription_client: tuple[AsyncClient, FakeSttResultClient],
) -> None:
    client, _ = transcription_client
    created = (await client.post("/v1/transcriptions", json=request_body())).json()
    app.dependency_overrides[get_current_user] = lambda: current_user(tenant_id="tenant-b")

    response = await client.post(
        f"/v1/transcriptions/{created['transcription_id']}/upload-complete",
        json={"etag": '"etag-1"', "size_bytes": 1024},
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_user_cannot_access_another_users_workflow(
    transcription_client: tuple[AsyncClient, FakeSttResultClient],
) -> None:
    client, _ = transcription_client
    created = (await client.post("/v1/transcriptions", json=request_body())).json()
    app.dependency_overrides[get_current_user] = lambda: current_user(user_id="user-b")

    status = await client.get(f"/v1/transcriptions/{created['transcription_id']}")
    complete = await client.post(
        f"/v1/transcriptions/{created['transcription_id']}/upload-complete",
        json={"etag": '"etag-1"', "size_bytes": 1024},
    )

    assert status.status_code == 404
    assert complete.status_code == 404
