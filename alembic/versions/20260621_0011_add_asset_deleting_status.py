"""add asset deleting workflow status

Revision ID: 20260621_0011
Revises: 20260621_0010
Create Date: 2026-06-21
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260621_0011"
down_revision: str | Sequence[str] | None = "20260621_0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TYPE transcription_workflow_status ADD VALUE IF NOT EXISTS 'deleting'")


def downgrade() -> None:
    pass
