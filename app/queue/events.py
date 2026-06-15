from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal, cast
from uuid import UUID

from pydantic import BaseModel, Field


class StreamEnvelope(BaseModel):
    event_id: str
    event_type: str
    schema_version: Literal["1.0"] = "1.0"
    payload: dict[str, Any]


class ProgressUpdated(BaseModel):
    transcription_id: UUID
    tenant_id: str
    status: str
    completed_chunks: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    failed_chunks: int = Field(ge=0)
    progress_ratio: float = Field(ge=0, le=1)
    occurred_at: datetime


class TranscriptCompleted(BaseModel):
    transcription_id: UUID
    tenant_id: str
    result_object_key: str


class TranscriptionCompleted(BaseModel):
    transcription_id: UUID
    tenant_id: str = Field(min_length=1, max_length=128)
    raw_text_sha256: str = Field(min_length=64, max_length=64)
    corrected_text_sha256: str = Field(min_length=64, max_length=64)
    dictionary_version: str = Field(min_length=1, max_length=32)
    result_object_key: str = Field(min_length=1)
    completed_at: datetime


def encode_envelope(envelope: StreamEnvelope) -> dict[str, str]:
    return {
        "event_id": envelope.event_id,
        "event_type": envelope.event_type,
        "schema_version": envelope.schema_version,
        "payload": json.dumps(envelope.payload, ensure_ascii=False, separators=(",", ":")),
    }


def decode_envelope(fields: dict[str, str]) -> StreamEnvelope:
    return StreamEnvelope(
        event_id=fields["event_id"],
        event_type=fields["event_type"],
        schema_version=cast(Literal["1.0"], fields.get("schema_version", "1.0")),
        payload=json.loads(fields["payload"]),
    )
