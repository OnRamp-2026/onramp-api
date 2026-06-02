"""Chat 서비스 — ChatRequest → LangGraph(compiled_graph) → ChatResponse."""

from __future__ import annotations

import logging

from app.agents.graph import compiled_graph
from app.agents.state import AnswerabilityStatus, Domain, FiveElements, SourceDocument
from app.config import get_settings
from app.middleware.error_handler import OnRampError
from app.models.request import ChatRequest
from app.models.response import ChatResponse, FiveElementsResponse, SourceDoc

logger = logging.getLogger(__name__)


async def chat(request: ChatRequest) -> ChatResponse:
    """질문을 그래프로 흘려 5요소 구조화 답변(ChatResponse)을 만든다."""
    settings = get_settings()
    initial_state = {
        "query": request.query,
        "model": request.model or settings.default_model,
    }
    try:
        state = await compiled_graph.ainvoke(initial_state)
    except OnRampError:
        raise  # 노드가 올린 도메인 에러(LLMError 502 등)는 그대로
    except Exception as exc:  # 그래프 실행 실패 → 500
        logger.exception("chat 파이프라인 실패")
        raise OnRampError("답변 생성 중 오류가 발생했습니다", status_code=500) from exc

    return _to_response(state, request)


def _to_response(state: dict, request: ChatRequest) -> ChatResponse:
    answer: FiveElements = state.get("answer") or FiveElements()
    # router가 UNANSWERABLE로 차단해 answer가 안 돈 경로는 status 미설정 → NOT_ENOUGH 기본
    status = state.get("answerability_status") or AnswerabilityStatus.NOT_ENOUGH_EVIDENCE
    domain = state.get("domain")
    return ChatResponse(
        answer=FiveElementsResponse(
            situation=answer.situation,
            cause=answer.cause,
            evidence=answer.evidence,
            solution=answer.solution,
            infra_context=answer.infra_context,
        ),
        sources=[_to_source(doc) for doc in state.get("sources", [])],
        answerability_status=status.value if isinstance(status, AnswerabilityStatus) else str(status),
        answerability_reason=state.get("answerability_reason", ""),
        domain=domain.value if isinstance(domain, Domain) else (domain or ""),
        model_used=state.get("model") or request.model,
    )


def _to_source(doc: SourceDocument) -> SourceDoc:
    return SourceDoc(
        title=doc.title,
        url=doc.url,
        space_key=doc.space_key,
        content_snippet=doc.content_snippet,
        score=doc.score,
    )
