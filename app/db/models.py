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
    ForeignKey,
    ForeignKeyConstraint,
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
    publishing = "publishing"
    published = "published"


class IndexRunType(StrEnum):
    full = "full"
    full_scan = "full_scan"
    incremental = "incremental"


class IndexRunStatus(StrEnum):
    queued = "queued"
    running = "running"
    success = "success"
    failed = "failed"


class IndexRunTrigger(StrEnum):
    manual = "manual"
    cron = "cron"


class IndexRunStage(StrEnum):
    queued = "queued"
    fetching = "fetching"
    preparing = "preparing"
    indexing = "indexing"
    completed = "completed"


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
    source_transcription_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("transcription_workflows.transcription_id", ondelete="CASCADE"),
        index=True,
    )
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
        Index("ix_report_jobs_status_created_at", "status", "created_at"),
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
    source_transcription_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("transcription_workflows.transcription_id", ondelete="CASCADE"),
        index=True,
    )
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
    confluence_page_id: Mapped[str] = mapped_column(String(128), default="")
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


class SourceDocument(Base):
    """멀티소스 인덱싱 원장 — confluence | github 등. (구 confluence_document)"""

    __tablename__ = "source_document"

    tenant_id: Mapped[str] = mapped_column(String(64), primary_key=True, default="onramp")
    # source를 PK에 포함 — confluence/github가 동일 page_id를 가져도 별도 레코드(덮어쓰기 방지).
    source: Mapped[str] = mapped_column(String(32), primary_key=True, default="confluence")  # confluence | github
    page_id: Mapped[str] = mapped_column(String(64), primary_key=True)  # 소스별 문서 id (gh:repo:path 등)
    space_key: Mapped[str] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(String(500))
    source_url: Mapped[str | None] = mapped_column(String(1000))
    domain: Mapped[str | None] = mapped_column(String(32))
    version: Mapped[str | None] = mapped_column(String(32))
    raw_html: Mapped[str | None] = mapped_column(Text)
    cleaned_markdown: Mapped[str | None] = mapped_column(Text)
    raw_html_hash: Mapped[str | None] = mapped_column(String(64))
    cleaned_markdown_hash: Mapped[str | None] = mapped_column(String(64))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    last_modified: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    __table_args__ = (
        UniqueConstraint("tenant_id", "source", "space_key", "page_id", name="uq_source_document_space_page"),
        Index("ix_source_document_domain", "tenant_id", "domain"),
        Index("ix_source_document_indexed_at", "tenant_id", "indexed_at"),
    )


class SourceDocumentPrevious(Base):
    """직전 snapshot — current/previous 1+1 정책. 새 버전 수집 시 현재본 → 여기로 회전. (구 confluence_document_previous)"""

    __tablename__ = "source_document_previous"

    tenant_id: Mapped[str] = mapped_column(String(64), primary_key=True, default="onramp")
    source: Mapped[str] = mapped_column(String(32), primary_key=True, default="confluence")
    page_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    space_key: Mapped[str] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(String(500))
    source_url: Mapped[str | None] = mapped_column(String(1000))
    domain: Mapped[str | None] = mapped_column(String(32))
    version: Mapped[str | None] = mapped_column(String(32))
    raw_html: Mapped[str | None] = mapped_column(Text)
    cleaned_markdown: Mapped[str | None] = mapped_column(Text)
    raw_html_hash: Mapped[str | None] = mapped_column(String(64))
    cleaned_markdown_hash: Mapped[str | None] = mapped_column(String(64))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    last_modified: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    replaced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "source", "page_id"],
            ["source_document.tenant_id", "source_document.source", "source_document.page_id"],
            name="fk_source_document_previous_current",
            ondelete="CASCADE",
        ),
    )


class IndexRun(Base):
    __tablename__ = "index_run"

    run_id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = mapped_column(String(64), default="onramp")
    run_type: Mapped[str] = mapped_column(String(16), default=IndexRunType.incremental.value)
    trigger: Mapped[str] = mapped_column(String(16), default=IndexRunTrigger.manual.value)
    stage: Mapped[str] = mapped_column(String(16), default=IndexRunStage.indexing.value)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    pages_discovered: Mapped[int] = mapped_column(Integer, default=0)
    pages_processed: Mapped[int] = mapped_column(Integer, default=0)
    pages_indexed: Mapped[int] = mapped_column(Integer, default=0)
    pages_skipped: Mapped[int] = mapped_column(Integer, default=0)
    pages_failed: Mapped[int] = mapped_column(Integer, default=0)
    chunks_indexed: Mapped[int] = mapped_column(Integer, default=0)
    chunks_deleted: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default=IndexRunStatus.running.value)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (Index("ix_index_run_tenant_status", "tenant_id", "status", created_at.desc()),)


class ChunkRegistry(Base):
    __tablename__ = "chunk_registry"

    chunk_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), default="onramp")
    point_id: Mapped[uuid.UUID]
    parent_id: Mapped[str] = mapped_column(String(80))
    page_id: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(32), default="confluence")  # 소속 문서의 source
    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("index_run.run_id", ondelete="SET NULL"))
    domain: Mapped[str | None] = mapped_column(String(32))
    section_type: Mapped[str | None] = mapped_column(String(40))
    token_count: Mapped[int | None] = mapped_column(Integer)
    hash: Mapped[str] = mapped_column(String(64))
    parent_content: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "source", "page_id"],
            ["source_document.tenant_id", "source_document.source", "source_document.page_id"],
            name="fk_chunk_registry_source_document",
            ondelete="CASCADE",
        ),
        Index("ix_chunk_registry_tenant_page", "tenant_id", "page_id"),
        Index("ix_chunk_registry_run_id", "run_id"),
        Index("ix_chunk_registry_point_id", "point_id", unique=True),
    )


class ChatLog(Base):
    __tablename__ = "chat_log"

    log_id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = mapped_column(String(64), default="onramp")
    query: Mapped[str] = mapped_column(Text)
    domain: Mapped[str | None] = mapped_column(String(32))
    use_case: Mapped[str | None] = mapped_column(String(16))
    answerability_status: Mapped[str | None] = mapped_column(String(32))
    answerability_reason: Mapped[str | None] = mapped_column(Text)
    model_used: Mapped[str | None] = mapped_column(String(64))
    source_count: Mapped[int] = mapped_column(Integer, default=0)
    sources: Mapped[dict[str, Any] | list[dict[str, Any]] | None] = mapped_column(JSON)
    latency_ms: Mapped[int | None] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        Index("ix_chat_log_tenant_created", "tenant_id", created_at.desc()),
        Index("ix_chat_log_domain", "tenant_id", "domain"),
    )


class Conversation(Base):
    """대화 1건 — 사이드바 '최근 대화' 목록의 한 줄. 로그인 사용자(tenant_id+user_id)에 귀속."""

    __tablename__ = "conversation"

    conversation_id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = mapped_column(String(64), default="onramp")
    user_id: Mapped[str] = mapped_column(String(128))
    title: Mapped[str] = mapped_column(String(500), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    __table_args__ = (
        # "내 최근 대화" 정렬 — 테넌트+유저 범위, 최신 갱신순
        Index("ix_conversation_tenant_user_updated", "tenant_id", "user_id", updated_at.desc()),
    )


class Message(Base):
    """대화 안의 한 턴(질문/답변). assistant 답변은 answer(5요소)+sources JSON으로 보관해 그대로 복원."""

    __tablename__ = "message"

    message_id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversation.conversation_id", ondelete="CASCADE"))
    tenant_id: Mapped[str] = mapped_column(String(64), default="onramp")
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, default="")
    answer: Mapped[dict[str, Any] | None] = mapped_column(JSON)  # assistant 5요소
    sources: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)  # 인용 출처 스냅샷
    domain: Mapped[str | None] = mapped_column(String(32))
    answerability_status: Mapped[str | None] = mapped_column(String(32))
    answerability_reason: Mapped[str | None] = mapped_column(Text)
    model_used: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (Index("ix_message_conversation_created", "conversation_id", "created_at"),)
