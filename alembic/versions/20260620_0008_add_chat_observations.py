"""add chat observations

Revision ID: 20260620_0008
Revises: 20260618_0007
Create Date: 2026-06-20 00:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260620_0008"
down_revision = "20260618_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_observations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("requested_model", sa.String(length=128), nullable=False),
        sa.Column("model_used", sa.String(length=128), nullable=False),
        sa.Column("domain", sa.String(length=64), nullable=False),
        sa.Column("answerability_status", sa.String(length=64), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("duration_ms", sa.BigInteger(), nullable=False),
        sa.Column("source_count", sa.Integer(), nullable=False),
        sa.Column("result_bucket", sa.String(length=16), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.Column("estimated_cost_usd", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chat_observations_request_id", "chat_observations", ["request_id"], unique=False)
    op.create_index("ix_chat_observations_result_bucket", "chat_observations", ["result_bucket"], unique=False)
    op.create_index("ix_chat_observations_tenant_id", "chat_observations", ["tenant_id"], unique=False)
    op.create_index(
        "ix_chat_observations_tenant_created",
        "chat_observations",
        ["tenant_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_chat_observations_tenant_bucket_created",
        "chat_observations",
        ["tenant_id", "result_bucket", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_chat_observations_tenant_bucket_created", table_name="chat_observations")
    op.drop_index("ix_chat_observations_tenant_created", table_name="chat_observations")
    op.drop_index("ix_chat_observations_tenant_id", table_name="chat_observations")
    op.drop_index("ix_chat_observations_result_bucket", table_name="chat_observations")
    op.drop_index("ix_chat_observations_request_id", table_name="chat_observations")
    op.drop_table("chat_observations")
