"""extend index_run for queued ingestion progress

Revision ID: 20260618_0007
Revises: 20260616_0006
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260618_0007"
down_revision: str | None = "20260616_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("index_run", sa.Column("trigger", sa.String(length=16), nullable=False, server_default="manual"))
    op.add_column("index_run", sa.Column("stage", sa.String(length=16), nullable=False, server_default="indexing"))
    op.add_column("index_run", sa.Column("pages_discovered", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("index_run", sa.Column("pages_processed", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("index_run", sa.Column("pages_skipped", sa.Integer(), nullable=False, server_default="0"))
    op.execute(
        "CREATE UNIQUE INDEX uq_index_run_active_tenant ON index_run (tenant_id) "
        "WHERE status IN ('queued', 'running')"
    )


def downgrade() -> None:
    op.drop_index("uq_index_run_active_tenant", table_name="index_run")
    op.drop_column("index_run", "pages_skipped")
    op.drop_column("index_run", "pages_processed")
    op.drop_column("index_run", "pages_discovered")
    op.drop_column("index_run", "stage")
    op.drop_column("index_run", "trigger")
