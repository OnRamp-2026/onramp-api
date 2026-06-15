from __future__ import annotations

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


class SttResultClient:
    def __init__(self, base_url: str, token: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds

    async def get_result(self, transcription_id: UUID) -> SttResult:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(
                f"{self.base_url}/v1/internal/transcriptions/{transcription_id}/result",
                headers=headers,
            )
        response.raise_for_status()
        return SttResult.model_validate(response.json())
