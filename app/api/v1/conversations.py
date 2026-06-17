"""GET /v1/conversations — 로그인 사용자별 대화 기록 조회."""

from fastapi import APIRouter, HTTPException, status

from app.api.deps import CurrentUser, DatabaseSession
from app.models.response import ConversationMessage, ConversationSummary, FiveElementsResponse, SourceDoc
from app.services.conversation_service import (
    delete_conversation,
    get_conversation_messages,
    list_conversations,
)

router = APIRouter()


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_my_conversations(user: CurrentUser, db: DatabaseSession) -> list[ConversationSummary]:
    """현재 사용자(테넌트+user)의 대화 목록을 최신 갱신순으로."""
    rows = await list_conversations(db, tenant_id=user.tenant_id, user_id=user.subject)
    return [
        ConversationSummary(
            conversation_id=str(c.conversation_id),
            title=c.title,
            updated_at=c.updated_at.isoformat(),
        )
        for c in rows
    ]


@router.get("/conversations/{conversation_id}/messages", response_model=list[ConversationMessage])
async def get_conversation(conversation_id: str, user: CurrentUser, db: DatabaseSession) -> list[ConversationMessage]:
    """대화 메시지를 시간순으로 복원. 소유자가 아니면 404."""
    messages = await get_conversation_messages(
        db, tenant_id=user.tenant_id, user_id=user.subject, conversation_id=conversation_id
    )
    if messages is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="대화를 찾을 수 없습니다.")
    return [
        ConversationMessage(
            role=m.role,
            content=m.content,
            answer=FiveElementsResponse(**m.answer) if m.answer else None,
            sources=[SourceDoc(**s) for s in (m.sources or [])],
            domain=m.domain or "",
            answerability_status=m.answerability_status or "",
            answerability_reason=m.answerability_reason or "",
            model_used=m.model_used or "",
            created_at=m.created_at.isoformat(),
        )
        for m in messages
    ]


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_my_conversation(conversation_id: str, user: CurrentUser, db: DatabaseSession) -> None:
    """본인 소유 대화를 삭제한다(message는 CASCADE). 소유자가 아니거나 없으면 404."""
    deleted = await delete_conversation(
        db, tenant_id=user.tenant_id, user_id=user.subject, conversation_id=conversation_id
    )
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="대화를 찾을 수 없습니다.")
