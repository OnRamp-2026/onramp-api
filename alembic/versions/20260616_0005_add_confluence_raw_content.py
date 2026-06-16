"""add raw_html/cleaned_markdown to confluence_document + confluence_document_previous

Revision ID: 20260616_0005
Revises: 20260616_0004
Create Date: 2026-06-16
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260616_0005"
down_revision: str | Sequence[str] | None = "20260616_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("confluence_document", sa.Column("raw_html", sa.Text(), nullable=True))
    op.add_column("confluence_document", sa.Column("cleaned_markdown", sa.Text(), nullable=True))

    op.create_table(
        "confluence_document_previous",
        sa.Column("page_id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False, server_default="onramp"),
        sa.Column("space_key", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("source_url", sa.String(length=1000), nullable=True),
        sa.Column("domain", sa.String(length=32), nullable=True),
        sa.Column("version", sa.String(length=32), nullable=True),
        sa.Column("raw_html", sa.Text(), nullable=True),
        sa.Column("cleaned_markdown", sa.Text(), nullable=True),
        sa.Column("raw_html_hash", sa.CHAR(length=64), nullable=True),
        sa.Column("cleaned_markdown_hash", sa.CHAR(length=64), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_modified", sa.DateTime(timezone=True), nullable=True),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("replaced_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("tenant_id", "page_id"),
        sa.ForeignKeyConstraint(
            ["tenant_id", "page_id"],
            ["confluence_document.tenant_id", "confluence_document.page_id"],
            ondelete="CASCADE",
        ),
    )


def downgrade() -> None:
    op.drop_table("confluence_document_previous")
    op.drop_column("confluence_document", "cleaned_markdown")
    op.drop_column("confluence_document", "raw_html")
