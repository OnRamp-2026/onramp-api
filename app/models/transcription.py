from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.db.models import WorkflowStatus


class TranscriptionCreateRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=512)
    content_type: str = Field(min_length=1, max_length=128)
    size_bytes: int = Field(gt=0)
    title: str = Field(default="", max_length=512)
    language: str = Field(default="ko-KR", min_length=1, max_length=32)
    category: str = Field(default="회의록", min_length=1, max_length=64)


class UploadCompleteRequest(BaseModel):
    etag: str = Field(min_length=1, max_length=256)
    size_bytes: int = Field(gt=0)


class UploadInstruction(BaseModel):
    method: Literal["PUT"] = "PUT"
    url: str
    headers: dict[str, str]
    expires_at: datetime


class TranscriptionCreateResponse(BaseModel):
    workflow_id: UUID
    transcription_id: UUID
    status: WorkflowStatus
    upload: UploadInstruction | None


class UploadCompleteResponse(BaseModel):
    transcription_id: UUID
    status: WorkflowStatus


class TranscriptionProgress(BaseModel):
    total_chunks: int
    completed_chunks: int
    failed_chunks: int
    percent: float


class ReportStatus(BaseModel):
    status: str = "not_started"
    report_id: UUID | None = None


class TranscriptionStatusResponse(BaseModel):
    transcription_id: UUID
    status: WorkflowStatus
    progress: TranscriptionProgress
    report: ReportStatus
    updated_at: datetime
