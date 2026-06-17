from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.queue.events import TranscriptCompleted, TranscriptionCompleted


def test_transcript_completed_rejects_cross_tenant_result_object_key() -> None:
    with pytest.raises(ValueError, match="result_object_key"):
        TranscriptCompleted(
            transcription_id=uuid4(),
            tenant_id="tenant-a",
            result_object_key="tenants/tenant-b/transcriptions/id/result/transcript.json",
        )


def test_transcription_completed_rejects_cross_tenant_result_object_key() -> None:
    with pytest.raises(ValueError, match="result_object_key"):
        TranscriptionCompleted(
            transcription_id=uuid4(),
            tenant_id="tenant-a",
            raw_text_sha256="a" * 64,
            corrected_text_sha256="b" * 64,
            dictionary_version="2026-06-16",
            result_object_key="tenants/tenant-b/transcriptions/id/result/corrected-transcript.json",
            completed_at=datetime.now(UTC),
        )
