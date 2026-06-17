from __future__ import annotations

from datetime import datetime
from uuid import UUID

import httpx
from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    start_time_sec: float
    end_time_sec: float
    text: str
    speaker: str | None = None
    confidence: float | None = None


class TranscriptResult(BaseModel):
    text_sha256: str
    text: str
    segments: list[TranscriptSegment]


class CorrectedTranscriptResult(TranscriptResult):
    correction_count: int
    review_candidate_count: int


class SttResult(BaseModel):
    schema_version: str = "1.0"
    transcription_id: UUID
    tenant_id: str
    provider: str
    audio_duration_sec: float | None
    dictionary_version: str
    raw: TranscriptResult
    corrected: CorrectedTranscriptResult


class SttUploadInstruction(BaseModel):
    url: str
    method: str
    headers: dict[str, str]
    expires_at: datetime


class SttCreateTranscriptionResponse(BaseModel):
    transcription_id: UUID
    status: str
    source_object_key: str
    upload: SttUploadInstruction


class SttCompleteUploadResponse(BaseModel):
    transcription_id: UUID
    status: str


class SttResultClient:
    def __init__(self, base_url: str, token: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

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
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/v1/internal/transcriptions",
                headers=self._headers,
                json={
                    "tenant_id": tenant_id,
                    "transcription_id": str(transcription_id),
                    "filename": filename,
                    "content_type": content_type,
                    "size_bytes": size_bytes,
                    "idempotency_key": idempotency_key,
                },
            )
        response.raise_for_status()
        return SttCreateTranscriptionResponse.model_validate(response.json())

    async def complete_upload(
        self,
        transcription_id: UUID,
        *,
        tenant_id: str,
        etag: str | None,
        size_bytes: int,
    ) -> SttCompleteUploadResponse:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/v1/internal/transcriptions/{transcription_id}/upload-complete",
                headers=self._headers,
                json={"tenant_id": tenant_id, "etag": etag, "size_bytes": size_bytes},
            )
        response.raise_for_status()
        return SttCompleteUploadResponse.model_validate(response.json())

    async def get_result(self, transcription_id: UUID) -> SttResult:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(
                f"{self.base_url}/v1/internal/transcriptions/{transcription_id}/result",
                headers=self._headers,
            )
        response.raise_for_status()
        return SttResult.model_validate(response.json())
