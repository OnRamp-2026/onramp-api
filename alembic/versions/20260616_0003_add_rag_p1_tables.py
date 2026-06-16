"""add RAG indexing P1 tables

Revision ID: 20260616_0003
Revises: 20260614_0002
Create Date: 2026-06-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260616_0003"
down_revision: str | Sequence[str] | None = "20260614_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "confluence_document",
        sa.Column("page_id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False, server_default="onramp"),
        sa.Column("space_key", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("source_url", sa.String(length=1000), nullable=True),
        sa.Column("domain", sa.String(length=32), nullable=True),
        sa.Column("version", sa.String(length=32), nullable=True),
        sa.Column("raw_html_hash", sa.CHAR(length=64), nullable=True),
        sa.Column("cleaned_markdown_hash", sa.CHAR(length=64), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_modified", sa.DateTime(timezone=True), nullable=True),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("tenant_id", "page_id"),
        sa.UniqueConstraint("tenant_id", "space_key", "page_id", name="uq_confluence_document_space_page"),
    )
    op.create_index("ix_confluence_document_domain", "confluence_document", ["tenant_id", "domain"])
    op.create_index("ix_confluence_document_indexed_at", "confluence_document", ["tenant_id", "indexed_at"])

    op.create_table(
        "index_run",
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False, server_default="onramp"),
        sa.Column("run_type", sa.String(length=16), nullable=False, server_default="incremental"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pages_indexed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("pages_failed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunks_indexed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunks_deleted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="running"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index("ix_index_run_tenant_status", "index_run", ["tenant_id", "status", sa.text("created_at DESC")])

    op.create_table(
        "chunk_registry",
        sa.Column("chunk_id", sa.String(length=80), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False, server_default="onramp"),
        sa.Column("point_id", sa.Uuid(), nullable=False),
        sa.Column("parent_id", sa.String(length=80), nullable=False),
        sa.Column("page_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=True),
        sa.Column("domain", sa.String(length=32), nullable=True),
        sa.Column("section_type", sa.String(length=40), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("hash", sa.CHAR(length=64), nullable=False),
        sa.Column("parent_content", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id", "page_id"],
            ["confluence_document.tenant_id", "confluence_document.page_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["index_run.run_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    op.create_index("ix_chunk_registry_tenant_page", "chunk_registry", ["tenant_id", "page_id"])
    op.create_index("ix_chunk_registry_run_id", "chunk_registry", ["run_id"])
    op.create_index("ix_chunk_registry_point_id", "chunk_registry", ["point_id"], unique=True)

    op.create_table(
        "chat_log",
        sa.Column("log_id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False, server_default="onramp"),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("domain", sa.String(length=32), nullable=True),
        sa.Column("use_case", sa.String(length=16), nullable=True),
        sa.Column("answerability_status", sa.String(length=32), nullable=True),
        sa.Column("answerability_reason", sa.Text(), nullable=True),
        sa.Column("model_used", sa.String(length=64), nullable=True),
        sa.Column("source_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("sources", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("latency_ms", sa.BigInteger(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("log_id"),
    )
    op.create_index("ix_chat_log_tenant_created", "chat_log", ["tenant_id", sa.text("created_at DESC")])
    op.create_index("ix_chat_log_domain", "chat_log", ["tenant_id", "domain"])


def downgrade() -> None:
    op.drop_index("ix_chat_log_domain", table_name="chat_log")
    op.drop_index("ix_chat_log_tenant_created", table_name="chat_log")
    op.drop_table("chat_log")

    op.drop_index("ix_chunk_registry_point_id", table_name="chunk_registry")
    op.drop_index("ix_chunk_registry_run_id", table_name="chunk_registry")
    op.drop_index("ix_chunk_registry_tenant_page", table_name="chunk_registry")
    op.drop_table("chunk_registry")

    op.drop_index("ix_index_run_tenant_status", table_name="index_run")
    op.drop_table("index_run")

    op.drop_index("ix_confluence_document_indexed_at", table_name="confluence_document")
    op.drop_index("ix_confluence_document_domain", table_name="confluence_document")
    op.drop_table("confluence_document")
