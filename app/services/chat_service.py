"""Chat 서비스 — ChatRequest → LangGraph(compiled_graph) → ChatResponse."""

from __future__ import annotations

import logging

from app.agents.graph import compiled_graph
from app.agents.state import AnswerabilityStatus, Domain, FiveElements, SourceDocument
from app.config import get_settings
from app.middleware.error_handler import OnRampError
from app.middleware.request_id import request_id_var
from app.models.request import ChatRequest
from app.models.response import ChatResponse, FiveElementsResponse, SourceDoc
from app.observability import current_trace_id, langfuse_run_config, langfuse_span, score_current_trace

logger = logging.getLogger(__name__)


async def chat(request: ChatRequest, tenant_id: str | None = None) -> ChatResponse:
    """질문을 그래프로 흘려 5요소 구조화 답변(ChatResponse)을 만든다."""
    # routing model에는 request.model만 — default_model을 섞으면 provider 선택이 그쪽으로 샌다.
    # 빈 model이면 selector가 config.llm_provider로 라우팅하고, default_model은 모델 이름으로만 쓰인다.
    settings = get_settings()
    initial_state = {
        "query": request.query,
        "model": request.model,
        "tenant_id": tenant_id or settings.auth_default_tenant,
        "retriever_strategy": settings.retriever_strategy,
        # Trust 재검색 루프 시드 — max_retries 한도로 무한루프 방지
        "retry_count": 0,
        "max_retries": settings.trust_max_retries,
    }
    # Langfuse 루트 span으로 한 턴을 감싼다 → 그래프(CallbackHandler) 노드 스팬 + call_llm
    # generation이 모두 이 한 trace 아래로 중첩된다(비활성이면 no-op). 노드 스팬 생성을 위해
    # CallbackHandler는 run_config로 계속 주입. tenant는 /chat 인증 연동(#98) 후 주입.
    with langfuse_span(name="chat", input={"query": request.query, "model": request.model}) as root:
        run_config = langfuse_run_config(request_id=request_id_var.get(), model=request.model)
        try:
            state = await compiled_graph.ainvoke(initial_state, config=run_config or None)
        except OnRampError:
            raise  # 노드가 올린 도메인 에러(LLMError 502 등)는 그대로
        except Exception as exc:  # 그래프 실행 실패 → 500
            logger.exception("chat 파이프라인 실패")
            raise OnRampError("답변 생성 중 오류가 발생했습니다", status_code=500) from exc

        response = _to_response(state, request)
        if root is not None:
            root.update(output={"answerability_status": response.answerability_status, "domain": response.domain})
            # trust_score를 online score로 부착 + trace_id를 응답에 노출(피드백 참조용)
            # state["trust_score"]는 TrustScore 객체 → overall([0,1] float)을 꺼낸다.
            overall = getattr(state.get("trust_score"), "overall", None)
            if isinstance(overall, int | float):
                score_current_trace(name="trust_score", value=float(overall))
            response.trace_id = current_trace_id() or ""
        return response


def _to_response(state: dict, request: ChatRequest) -> ChatResponse:
    """그래프 최종 AgentState를 ChatResponse로 매핑한다 (status 미설정 시 NOT_ENOUGH 기본)."""
    answer: FiveElements = state.get("answer") or FiveElements()
    # router가 UNANSWERABLE로 차단해 answer가 안 돈 경로는 status 미설정 → NOT_ENOUGH 기본
    status = state.get("answerability_status") or AnswerabilityStatus.NOT_ENOUGH_EVIDENCE
    domain = state.get("domain")
    return ChatResponse(
        answer_format=state.get("answer_format") or "structured",
        answer=FiveElementsResponse(
            situation=answer.situation,
            cause=answer.cause,
            evidence=answer.evidence,
            solution=answer.solution,
            infra_context=answer.infra_context,
        ),
        answer_text=state.get("answer_text", ""),
        sources=[_to_source(doc) for doc in state.get("sources", [])],
        answerability_status=status.value if isinstance(status, AnswerabilityStatus) else str(status),
        answerability_reason=state.get("answerability_reason", ""),
        domain=domain.value if isinstance(domain, Domain) else (domain or ""),
        model_used=state.get("model") or request.model or get_settings().default_model,
    )


def _to_source(doc: SourceDocument) -> SourceDoc:
    """내부 SourceDocument를 응답용 SourceDoc로 변환한다."""
    return SourceDoc(
        title=doc.title,
        url=doc.url,
        space_key=doc.space_key,
        content_snippet=doc.content_snippet,
        score=doc.score,
        site=doc.site,
        product_version=doc.product_version,
    )
