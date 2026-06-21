"""add asset workflow owner

Revision ID: 20260621_0010
Revises: 20260621_0009
Create Date: 2026-06-21
"""

import sqlalchemy as sa

from alembic import op

revision = "20260621_0010"
down_revision = "20260621_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "transcription_workflows",
        sa.Column("created_by_user_id", sa.String(length=128), server_default="", nullable=False),
    )
    op.drop_constraint(
        "uq_transcription_workflow_idempotency",
        "transcription_workflows",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_transcription_workflow_user_idempotency",
        "transcription_workflows",
        ["tenant_id", "created_by_user_id", "idempotency_key"],
    )
    op.create_index(
        "ix_transcription_workflow_tenant_user_updated_at",
        "transcription_workflows",
        ["tenant_id", "created_by_user_id", sa.text("updated_at DESC")],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_transcription_workflow_tenant_user_updated_at",
        table_name="transcription_workflows",
    )
    op.drop_constraint(
        "uq_transcription_workflow_user_idempotency",
        "transcription_workflows",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_transcription_workflow_idempotency",
        "transcription_workflows",
        ["tenant_id", "idempotency_key"],
    )
    op.drop_column("transcription_workflows", "created_by_user_id")
