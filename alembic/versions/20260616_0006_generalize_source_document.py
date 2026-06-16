"""generalize confluence_document → source_document (멀티소스 인덱싱 원장)

confluence_document / confluence_document_previous를 source_document(_previous)로 rename하고
``source`` 컬럼(confluence|github)을 **식별키(PK)** 에 포함한다. confluence/github가 동일 page_id를
가져도 별도 레코드로 취급되어 덮어쓰기·오참조가 발생하지 않는다 (CodeRabbit #177 Major).
데이터 보존(rename, drop X). 기존 행은 source='confluence'.

주의: 0003/0005가 FK를 **이름 없이** 생성해 DB 실제명은 Postgres 자동생성명이다
(chunk_registry_tenant_id_page_id_fkey, confluence_document_previous_tenant_id_page_id_fkey).
PK 변경을 위해 자식 FK를 먼저 떼고, rename 후 source 포함 FK를 모델과 동일한 이름으로 재생성한다.

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
    # 1) source 컬럼 추가(기존 행 server_default 'confluence' 백필) → app-default로 전환(default 제거)
    for table in ("confluence_document", "confluence_document_previous", "chunk_registry"):
        op.add_column(table, sa.Column("source", sa.String(length=32), nullable=False, server_default="confluence"))
        op.alter_column(table, "source", existing_type=sa.String(length=32), server_default=None)

    # 2) PK 변경 전, source_document(전신)를 참조하는 자식 FK 제거 (0003/0005 자동생성명)
    op.drop_constraint(
        "confluence_document_previous_tenant_id_page_id_fkey", "confluence_document_previous", type_="foreignkey"
    )
    op.drop_constraint("chunk_registry_tenant_id_page_id_fkey", "chunk_registry", type_="foreignkey")

    # 3) PK·unique 제거 (rename 후 source 포함으로 재생성)
    op.drop_constraint("confluence_document_pkey", "confluence_document", type_="primary")
    op.drop_constraint("confluence_document_previous_pkey", "confluence_document_previous", type_="primary")
    op.drop_constraint("uq_confluence_document_space_page", "confluence_document", type_="unique")

    # 4) 테이블 rename (데이터 보존)
    op.rename_table("confluence_document", "source_document")
    op.rename_table("confluence_document_previous", "source_document_previous")
    op.execute("ALTER INDEX ix_confluence_document_domain RENAME TO ix_source_document_domain")
    op.execute("ALTER INDEX ix_confluence_document_indexed_at RENAME TO ix_source_document_indexed_at")

    # 5) PK·unique·FK를 source 포함으로 재생성 (모델 이름과 일치)
    op.create_primary_key("source_document_pkey", "source_document", ["tenant_id", "source", "page_id"])
    op.create_primary_key(
        "source_document_previous_pkey", "source_document_previous", ["tenant_id", "source", "page_id"]
    )
    op.create_unique_constraint(
        "uq_source_document_space_page", "source_document", ["tenant_id", "source", "space_key", "page_id"]
    )
    op.create_foreign_key(
        "fk_source_document_previous_current",
        "source_document_previous",
        "source_document",
        ["tenant_id", "source", "page_id"],
        ["tenant_id", "source", "page_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_chunk_registry_source_document",
        "chunk_registry",
        "source_document",
        ["tenant_id", "source", "page_id"],
        ["tenant_id", "source", "page_id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    # source 포함 FK·PK·unique 제거 → rename 복원 → 원래(이름 없는) FK·PK·unique 재생성 → source 컬럼 제거
    op.drop_constraint("fk_chunk_registry_source_document", "chunk_registry", type_="foreignkey")
    op.drop_constraint("fk_source_document_previous_current", "source_document_previous", type_="foreignkey")
    op.drop_constraint("uq_source_document_space_page", "source_document", type_="unique")
    op.drop_constraint("source_document_previous_pkey", "source_document_previous", type_="primary")
    op.drop_constraint("source_document_pkey", "source_document", type_="primary")

    op.execute("ALTER INDEX ix_source_document_indexed_at RENAME TO ix_confluence_document_indexed_at")
    op.execute("ALTER INDEX ix_source_document_domain RENAME TO ix_confluence_document_domain")
    op.rename_table("source_document_previous", "confluence_document_previous")
    op.rename_table("source_document", "confluence_document")

    op.create_primary_key("confluence_document_pkey", "confluence_document", ["tenant_id", "page_id"])
    op.create_primary_key(
        "confluence_document_previous_pkey", "confluence_document_previous", ["tenant_id", "page_id"]
    )
    op.create_unique_constraint(
        "uq_confluence_document_space_page", "confluence_document", ["tenant_id", "space_key", "page_id"]
    )
    # 0003/0005와 동일하게 이름 없는 FK로 복원 (Postgres 자동생성명)
    op.create_foreign_key(
        None,
        "confluence_document_previous",
        "confluence_document",
        ["tenant_id", "page_id"],
        ["tenant_id", "page_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        None,
        "chunk_registry",
        "confluence_document",
        ["tenant_id", "page_id"],
        ["tenant_id", "page_id"],
        ondelete="CASCADE",
    )

    op.drop_column("chunk_registry", "source")
    op.drop_column("confluence_document_previous", "source")
    op.drop_column("confluence_document", "source")
