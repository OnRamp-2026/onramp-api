"""add STT event inbox, report jobs, and reports

Revision ID: 20260614_0002
Revises: 20260612_0001
Create Date: 2026-06-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260614_0002"
down_revision: str | Sequence[str] | None = "20260612_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

report_job_status = postgresql.ENUM(
    "queued",
    "processing",
    "completed",
    "failed",
    name="report_job_status",
    create_type=False,
)
report_status = postgresql.ENUM("draft", "publishing", "published", name="report_status", create_type=False)


def upgrade() -> None:
    report_job_status.create(op.get_bind(), checkfirst=True)
    report_status.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "event_inbox",
        sa.Column("consumer_group", sa.String(length=128), nullable=False),
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column("result_reference", sa.String(length=128), nullable=True),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("consumer_group", "event_id"),
    )
    op.create_table(
        "report_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("source_transcription_id", sa.Uuid(), nullable=False),
        sa.Column("status", report_job_status, nullable=False),
        sa.Column("raw_text_sha256", sa.String(length=64), nullable=False),
        sa.Column("corrected_text_sha256", sa.String(length=64), nullable=False),
        sa.Column("dictionary_version", sa.String(length=32), nullable=False),
        sa.Column("result_object_key", sa.Text(), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_transcription_id"],
            ["transcription_workflows.transcription_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "source_transcription_id",
            name="uq_report_job_source_transcription",
        ),
    )
    op.create_index("ix_report_jobs_tenant_id", "report_jobs", ["tenant_id"])
    op.create_index("ix_report_jobs_source_transcription_id", "report_jobs", ["source_transcription_id"])
    op.create_index(
        "ix_report_jobs_status_created_at",
        "report_jobs",
        ["status", "created_at"],
    )
    op.create_table(
        "reports",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("source_transcription_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("situation", sa.Text(), nullable=False),
        sa.Column("cause", sa.Text(), nullable=False),
        sa.Column("evidence", sa.Text(), nullable=False),
        sa.Column("solution", sa.Text(), nullable=False),
        sa.Column("infra_context", sa.Text(), nullable=False),
        sa.Column("status", report_status, nullable=False),
        sa.Column("raw_text_sha256", sa.String(length=64), nullable=False),
        sa.Column("corrected_text_sha256", sa.String(length=64), nullable=False),
        sa.Column("dictionary_version", sa.String(length=32), nullable=False),
        sa.Column("result_object_key", sa.Text(), nullable=False),
        sa.Column("confluence_page_id", sa.String(length=128), nullable=False),
        sa.Column("confluence_url", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_transcription_id"],
            ["transcription_workflows.transcription_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "source_transcription_id", name="uq_report_source_transcription"),
    )
    op.create_index("ix_reports_tenant_id", "reports", ["tenant_id"])
    op.create_index("ix_reports_source_transcription_id", "reports", ["source_transcription_id"])


def downgrade() -> None:
    op.drop_index("ix_reports_source_transcription_id", table_name="reports")
    op.drop_index("ix_reports_tenant_id", table_name="reports")
    op.drop_table("reports")
    op.drop_index("ix_report_jobs_status_created_at", table_name="report_jobs")
    op.drop_index("ix_report_jobs_source_transcription_id", table_name="report_jobs")
    op.drop_index("ix_report_jobs_tenant_id", table_name="report_jobs")
    op.drop_table("report_jobs")
    op.drop_table("event_inbox")
    report_status.drop(op.get_bind(), checkfirst=True)
    report_job_status.drop(op.get_bind(), checkfirst=True)
