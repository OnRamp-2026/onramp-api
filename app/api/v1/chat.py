"""POST /v1/chat 엔드포인트."""

from fastapi import APIRouter

from app.models.request import ChatRequest, FeedbackRequest
from app.models.response import ChatResponse
from app.observability import create_trace_score
from app.services.chat_service import chat as chat_service

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """자연어 질문 → Router → Retriever → Answer → 5요소 구조화 답변."""
    return await chat_service(request)


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
