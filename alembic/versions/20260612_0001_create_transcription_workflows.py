"""create transcription workflows and event outbox

Revision ID: 20260612_0001
Revises:
Create Date: 2026-06-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260612_0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

workflow_status = postgresql.ENUM(
    "awaiting_upload",
    "queued",
    "preprocessing",
    "transcribing",
    "merging",
    "transcript_completed",
    "correcting",
    "correction_completed",
    "report_queued",
    "report_processing",
    "draft",
    "published",
    "transcription_failed",
    "correction_failed",
    "report_failed",
    "cancelled",
    name="transcription_workflow_status",
    create_type=False,
)


def upgrade() -> None:
    workflow_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "transcription_workflows",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("transcription_id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=True),
        sa.Column("status", workflow_status, nullable=False),
        sa.Column("source_object_key", sa.Text(), nullable=False),
        sa.Column("source_filename", sa.String(length=512), nullable=False),
        sa.Column("source_content_type", sa.String(length=128), nullable=False),
        sa.Column("source_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("source_etag", sa.String(length=256), nullable=True),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("language", sa.String(length=32), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("total_chunks", sa.Integer(), nullable=False),
        sa.Column("completed_chunks", sa.Integer(), nullable=False),
        sa.Column("failed_chunks", sa.Integer(), nullable=False),
        sa.Column("transcript_completed_received_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("report_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "idempotency_key", name="uq_transcription_workflow_idempotency"),
        sa.UniqueConstraint("tenant_id", "transcription_id", name="uq_transcription_workflow_tenant"),
        sa.UniqueConstraint("transcription_id"),
    )
    op.create_index(
        "ix_transcription_workflows_transcription_id",
        "transcription_workflows",
        ["transcription_id"],
    )
    op.create_index("ix_transcription_workflows_tenant_id", "transcription_workflows", ["tenant_id"])

    op.create_table(
        "event_outbox",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("aggregate_type", sa.String(length=64), nullable=False),
        sa.Column("aggregate_id", sa.String(length=128), nullable=False),
        sa.Column("event_type", sa.String(length=128), nullable=False),
        sa.Column("stream_name", sa.String(length=128), nullable=False),
        sa.Column("payload_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("publish_attempts", sa.Integer(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_event_outbox_aggregate_id", "event_outbox", ["aggregate_id"])
    op.create_index("ix_event_outbox_available_at", "event_outbox", ["available_at"])
    op.create_index(
        "ix_event_outbox_pending",
        "event_outbox",
        ["available_at", "created_at"],
        postgresql_where=sa.text("published_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_event_outbox_pending", table_name="event_outbox")
    op.drop_index("ix_event_outbox_available_at", table_name="event_outbox")
    op.drop_index("ix_event_outbox_aggregate_id", table_name="event_outbox")
    op.drop_table("event_outbox")
    op.drop_index("ix_transcription_workflows_tenant_id", table_name="transcription_workflows")
    op.drop_index("ix_transcription_workflows_transcription_id", table_name="transcription_workflows")
    op.drop_table("transcription_workflows")
    workflow_status.drop(op.get_bind(), checkfirst=True)
