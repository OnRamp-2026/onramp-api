"""add conversation history tables

Revision ID: 20260616_0004
Revises: 20260616_0003
Create Date: 2026-06-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260616_0004"
down_revision: str | Sequence[str] | None = "20260616_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "conversation",
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False, server_default="onramp"),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("conversation_id"),
    )
    op.create_index(
        "ix_conversation_tenant_user_updated",
        "conversation",
        ["tenant_id", "user_id", sa.text("updated_at DESC")],
    )

    op.create_table(
        "message",
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False, server_default="onramp"),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("answer", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("sources", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("domain", sa.String(length=32), nullable=True),
        sa.Column("answerability_status", sa.String(length=32), nullable=True),
        sa.Column("answerability_reason", sa.Text(), nullable=True),
        sa.Column("model_used", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["conversation.conversation_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("message_id"),
    )
    op.create_index("ix_message_conversation_created", "message", ["conversation_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_message_conversation_created", table_name="message")
    op.drop_table("message")

    op.drop_index("ix_conversation_tenant_user_updated", table_name="conversation")
    op.drop_table("conversation")
