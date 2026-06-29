"""message에 answer_format·answer_text 추가 — 히스토리 복원 시 freeform/structured 렌더 분기 (#191)

기존 message는 answer(5요소 JSON)만 보관해 freeform 답변의 본문·포맷이 유실됐다.
히스토리도 라이브 채팅과 동일하게 렌더하려면 answer_format·answer_text가 필요하다.
기존 행은 structured로 백필(server_default) 후 app-default로 전환.

Revision ID: 20260621_0009
Revises: 20260620_0008
Create Date: 2026-06-21
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260621_0009"
down_revision: str | Sequence[str] | None = "20260620_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "message",
        sa.Column("answer_format", sa.String(length=16), nullable=False, server_default="structured"),
    )
    op.add_column("message", sa.Column("answer_text", sa.Text(), nullable=False, server_default=""))
    # 백필 끝 → app-default로 전환(server_default 제거)
    op.alter_column("message", "answer_format", existing_type=sa.String(length=16), server_default=None)
    op.alter_column("message", "answer_text", existing_type=sa.Text(), server_default=None)


def downgrade() -> None:
    op.drop_column("message", "answer_text")
    op.drop_column("message", "answer_format")
