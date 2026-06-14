from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Enum,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class WorkflowStatus(StrEnum):
    awaiting_upload = "awaiting_upload"
    queued = "queued"
    preprocessing = "preprocessing"
    transcribing = "transcribing"
    merging = "merging"
    transcript_completed = "transcript_completed"
    correcting = "correcting"
    correction_completed = "correction_completed"
    report_queued = "report_queued"
    report_processing = "report_processing"
    draft = "draft"
    published = "published"
    transcription_failed = "transcription_failed"
    correction_failed = "correction_failed"
    report_failed = "report_failed"
    cancelled = "cancelled"


class ReportJobStatus(StrEnum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class ReportStatus(StrEnum):
    draft = "draft"
    published = "published"


class TranscriptionWorkflow(Base):
    __tablename__ = "transcription_workflows"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    transcription_id: Mapped[uuid.UUID] = mapped_column(unique=True, index=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[WorkflowStatus] = mapped_column(
        Enum(
            WorkflowStatus,
            name="transcription_workflow_status",
            values_callable=lambda enum: [item.value for item in enum],
        ),
        default=WorkflowStatus.awaiting_upload,
    )
    source_object_key: Mapped[str] = mapped_column(Text)
    source_filename: Mapped[str] = mapped_column(String(512))
    source_content_type: Mapped[str] = mapped_column(String(128))
    source_size_bytes: Mapped[int] = mapped_column(BigInteger)
    source_etag: Mapped[str | None] = mapped_column(String(256))
    title: Mapped[str] = mapped_column(String(512))
    language: Mapped[str] = mapped_column(String(32), default="ko-KR")
    category: Mapped[str] = mapped_column(String(64))
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    completed_chunks: Mapped[int] = mapped_column(Integer, default=0)
    failed_chunks: Mapped[int] = mapped_column(Integer, default=0)
    transcript_completed_received_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    report_id: Mapped[uuid.UUID | None]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    __table_args__ = (
        UniqueConstraint("tenant_id", "idempotency_key", name="uq_transcription_workflow_idempotency"),
        UniqueConstraint("tenant_id", "transcription_id", name="uq_transcription_workflow_tenant"),
    )


class EventOutbox(Base):
    __tablename__ = "event_outbox"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    aggregate_type: Mapped[str] = mapped_column(String(64))
    aggregate_id: Mapped[str] = mapped_column(String(128), index=True)
    event_type: Mapped[str] = mapped_column(String(128))
    stream_name: Mapped[str] = mapped_column(String(128))
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    publish_attempts: Mapped[int] = mapped_column(Integer, default=0)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        Index(
            "ix_event_outbox_pending",
            "available_at",
            "created_at",
            postgresql_where=published_at.is_(None),
        ),
    )


class EventInbox(Base):
    __tablename__ = "event_inbox"

    consumer_group: Mapped[str] = mapped_column(String(128), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    result_reference: Mapped[str | None] = mapped_column(String(128))
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReportJob(Base):
    __tablename__ = "report_jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    source_transcription_id: Mapped[uuid.UUID] = mapped_column(index=True)
    status: Mapped[ReportJobStatus] = mapped_column(
        Enum(
            ReportJobStatus,
            name="report_job_status",
            values_callable=lambda enum: [item.value for item in enum],
        ),
        default=ReportJobStatus.queued,
    )
    raw_text_sha256: Mapped[str] = mapped_column(String(64))
    corrected_text_sha256: Mapped[str] = mapped_column(String(64))
    dictionary_version: Mapped[str] = mapped_column(String(32))
    result_object_key: Mapped[str] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "source_transcription_id",
            name="uq_report_job_source_transcription",
        ),
    )


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    source_transcription_id: Mapped[uuid.UUID] = mapped_column(index=True)
    title: Mapped[str] = mapped_column(String(512))
    category: Mapped[str] = mapped_column(String(64))
    situation: Mapped[str] = mapped_column(Text, default="")
    cause: Mapped[str] = mapped_column(Text, default="")
    evidence: Mapped[str] = mapped_column(Text, default="")
    solution: Mapped[str] = mapped_column(Text, default="")
    infra_context: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[ReportStatus] = mapped_column(
        Enum(
            ReportStatus,
            name="report_status",
            values_callable=lambda enum: [item.value for item in enum],
        ),
        default=ReportStatus.draft,
    )
    raw_text_sha256: Mapped[str] = mapped_column(String(64))
    corrected_text_sha256: Mapped[str] = mapped_column(String(64))
    dictionary_version: Mapped[str] = mapped_column(String(32))
    result_object_key: Mapped[str] = mapped_column(Text)
    confluence_url: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "source_transcription_id",
            name="uq_report_source_transcription",
        ),
    )
