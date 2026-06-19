"""POST /v1/chat 엔드포인트."""

import logging

from fastapi import APIRouter

from app.api.deps import CurrentUser, DatabaseSession
from app.models.request import ChatRequest, FeedbackRequest
from app.models.response import ChatResponse
from app.observability import create_trace_score
from app.services.chat_service import chat as chat_service
from app.services.conversation_service import persist_turn

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, user: CurrentUser, db: DatabaseSession) -> ChatResponse:
    """자연어 질문 → Router → Retriever → Answer → 5요소 구조화 답변.

    인증 필수(#163) — 미인증 요청은 401. 질문/답변을 사용자 대화 기록에 저장하고
    conversation_id를 응답에 실어준다.
    """
    response = await chat_service(request, tenant_id=user.tenant_id)
    if user.subject:
        try:
            response.conversation_id = await persist_turn(
                db, tenant_id=user.tenant_id, user_id=user.subject, request=request, response=response
            )
        except Exception:  # 저장 실패가 답변 응답을 막지 않도록
            logger.exception("대화 기록 저장 실패")
    return response


@router.post("/chat/feedback")
async def chat_feedback(request: FeedbackRequest) -> dict[str, bool]:
    """답변 trace에 사용자 피드백(👍/👎)을 Langfuse score로 기록한다.

    관측 비활성이면 best-effort no-op(recorded=false). 응답 흐름은 막지 않는다.
    """
    recorded = create_trace_score(
        trace_id=request.trace_id,
        name="user_feedback",
        value=request.value,
        comment=request.comment or None,
    )
    return {"recorded": recorded}
