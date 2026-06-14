from __future__ import annotations

from typing import Any
from uuid import uuid4

import httpx
import pytest

from app.services.stt_result_client import SttResultClient


@pytest.mark.asyncio
async def test_stt_result_client_uses_service_v1_route(monkeypatch: pytest.MonkeyPatch) -> None:
    transcription_id = uuid4()
    requested_urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(
            200,
            request=request,
            json={
                "schema_version": "1.0",
                "transcription_id": str(transcription_id),
                "tenant_id": "tenant-a",
                "provider": "clova",
                "audio_duration_sec": 10.0,
                "dictionary_version": "2026-06-14",
                "raw": {"text_sha256": "a" * 64, "text": "raw", "segments": []},
                "corrected": {
                    "text_sha256": "b" * 64,
                    "text": "corrected",
                    "segments": [],
                    "correction_count": 1,
                    "review_candidate_count": 0,
                },
            },
        )

    class MockAsyncClient(httpx.AsyncClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    await SttResultClient("http://stt-api:8000", "").get_result(transcription_id)

    assert requested_urls == [
        f"http://stt-api:8000/v1/internal/transcriptions/{transcription_id}/result",
    ]
