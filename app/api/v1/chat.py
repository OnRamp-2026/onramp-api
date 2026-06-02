"""POST /v1/chat 엔드포인트."""

from fastapi import APIRouter

from app.models.request import ChatRequest
from app.models.response import ChatResponse
from app.services.chat_service import chat as chat_service

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """자연어 질문 → Router → Retriever → Answer → 5요소 구조화 답변."""
    return await chat_service(request)
