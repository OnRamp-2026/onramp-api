"""generalize confluence_document → source_document (멀티소스 인덱싱 원장)

confluence_document / confluence_document_previous를 source_document(_previous)로 rename하고
``source`` 컬럼(confluence|github)을 추가한다. 데이터 보존(rename, drop X). 기존 행은 source='confluence'.
chunk_registry / previous FK는 Postgres가 rename을 자동 추종하므로 제약 '이름'만 갱신.

Revision ID: 20260616_0006
Revises: 20260616_0005
Create Date: 2026-06-16
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260616_0006"
down_revision: str | Sequence[str] | None = "20260616_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1) source 컬럼 추가(기존 행은 server_default로 'confluence' 백필) → server_default 제거(모델은 app-default)
    for table in ("confluence_document", "confluence_document_previous"):
        op.add_column(table, sa.Column("source", sa.String(length=32), nullable=False, server_default="confluence"))
        op.alter_column(table, "source", existing_type=sa.String(length=32), server_default=None)

    # 2) 테이블 rename (데이터 보존)
    op.rename_table("confluence_document", "source_document")
    op.rename_table("confluence_document_previous", "source_document_previous")

    # 3) 제약/인덱스 이름 갱신 (FK 참조는 OID라 자동 추종, 이름만)
    op.execute("ALTER TABLE source_document RENAME CONSTRAINT uq_confluence_document_space_page TO uq_source_document_space_page")
    op.execute("ALTER INDEX ix_confluence_document_domain RENAME TO ix_source_document_domain")
    op.execute("ALTER INDEX ix_confluence_document_indexed_at RENAME TO ix_source_document_indexed_at")
    op.execute(
        "ALTER TABLE source_document_previous "
        "RENAME CONSTRAINT fk_confluence_document_previous_current TO fk_source_document_previous_current"
    )
    op.execute(
        "ALTER TABLE chunk_registry "
        "RENAME CONSTRAINT fk_chunk_registry_confluence_document TO fk_chunk_registry_source_document"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE chunk_registry "
        "RENAME CONSTRAINT fk_chunk_registry_source_document TO fk_chunk_registry_confluence_document"
    )
    op.execute(
        "ALTER TABLE source_document_previous "
        "RENAME CONSTRAINT fk_source_document_previous_current TO fk_confluence_document_previous_current"
    )
    op.execute("ALTER INDEX ix_source_document_indexed_at RENAME TO ix_confluence_document_indexed_at")
    op.execute("ALTER INDEX ix_source_document_domain RENAME TO ix_confluence_document_domain")
    op.execute("ALTER TABLE source_document RENAME CONSTRAINT uq_source_document_space_page TO uq_confluence_document_space_page")

    op.rename_table("source_document_previous", "confluence_document_previous")
    op.rename_table("source_document", "confluence_document")

    op.drop_column("confluence_document_previous", "source")
    op.drop_column("confluence_document", "source")
