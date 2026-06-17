"""대화 기록(conversation/message) 저장·조회 서비스.

로그인 사용자(tenant_id+user_id)에 귀속된 대화를 영속화한다. chat_log(분석/관측)와 별개로,
사이드바 '최근 대화'와 대화 복원에 쓰이는 사용자 대면 저장소.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, cast

from sqlalchemy import CursorResult, delete, select

from app.db.models import Conversation, Message

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.models.request import ChatRequest
    from app.models.response import ChatResponse

_TITLE_MAX = 80


def _make_title(query: str) -> str:
    title = " ".join(query.strip().split())
    return title[:_TITLE_MAX] if title else "새 대화"


def _coerce_uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return None


async def persist_turn(
    db: AsyncSession,
    *,
    tenant_id: str,
    user_id: str,
    request: ChatRequest,
    response: ChatResponse,
) -> str:
    """질문/답변 한 턴을 저장한다. conversation_id가 없거나 소유자가 아니면 새 대화를 만든다.

    반환: 저장된 대화 ID(문자열). 트랜잭션 커밋까지 수행.
    """
    conversation = await _resolve_conversation(
        db, tenant_id=tenant_id, user_id=user_id, conversation_id=request.conversation_id, query=request.query
    )

    db.add(
        Message(
            conversation_id=conversation.conversation_id,
            tenant_id=tenant_id,
            role="user",
            content=request.query,
        )
    )
    db.add(
        Message(
            conversation_id=conversation.conversation_id,
            tenant_id=tenant_id,
            role="assistant",
            content=response.answer.situation or "",
            answer=response.answer.model_dump(),
            sources=[s.model_dump() for s in response.sources],
            domain=response.domain or None,
            answerability_status=response.answerability_status or None,
            answerability_reason=response.answerability_reason or None,
            model_used=response.model_used or None,
        )
    )
    # updated_at 갱신(목록 정렬) — onupdate는 flush 시 적용되므로 명시 dirty 표시
    conversation.title = conversation.title or _make_title(request.query)
    await db.flush()
    await db.commit()
    return str(conversation.conversation_id)


async def _resolve_conversation(
    db: AsyncSession, *, tenant_id: str, user_id: str, conversation_id: str, query: str
) -> Conversation:
    cid = _coerce_uuid(conversation_id)
    if cid is not None:
        existing = await db.scalar(
            select(Conversation).where(
                Conversation.conversation_id == cid,
                Conversation.tenant_id == tenant_id,
                Conversation.user_id == user_id,
            )
        )
        if existing is not None:
            return existing

    conversation = Conversation(tenant_id=tenant_id, user_id=user_id, title=_make_title(query))
    db.add(conversation)
    await db.flush()
    return conversation


async def list_conversations(db: AsyncSession, *, tenant_id: str, user_id: str, limit: int = 50) -> list[Conversation]:
    result = await db.scalars(
        select(Conversation)
        .where(Conversation.tenant_id == tenant_id, Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
    )
    return list(result)


async def get_conversation_messages(
    db: AsyncSession, *, tenant_id: str, user_id: str, conversation_id: str
) -> list[Message] | None:
    """대화 메시지를 시간순으로. 대화가 없거나 소유자가 아니면 None."""
    cid = _coerce_uuid(conversation_id)
    if cid is None:
        return None
    owner = await db.scalar(
        select(Conversation.conversation_id).where(
            Conversation.conversation_id == cid,
            Conversation.tenant_id == tenant_id,
            Conversation.user_id == user_id,
        )
    )
    if owner is None:
        return None
    result = await db.scalars(select(Message).where(Message.conversation_id == cid).order_by(Message.created_at))
    return list(result)


async def delete_conversation(db: AsyncSession, *, tenant_id: str, user_id: str, conversation_id: str) -> bool:
    """본인 소유 대화를 삭제한다. message는 FK ON DELETE CASCADE로 함께 제거.

    반환: 삭제됐으면 True, 대화가 없거나 소유자가 아니면 False.
    """
    cid = _coerce_uuid(conversation_id)
    if cid is None:
        return False
    # 소유권 조건을 DELETE WHERE에 직접 포함 — 검증+삭제 원자적(TOCTOU 경쟁 제거). rowcount로 실삭제 판단.
    result = await db.execute(
        delete(Conversation).where(
            Conversation.conversation_id == cid,
            Conversation.tenant_id == tenant_id,
            Conversation.user_id == user_id,
        )
    )
    await db.commit()
    return cast(CursorResult, result).rowcount > 0
