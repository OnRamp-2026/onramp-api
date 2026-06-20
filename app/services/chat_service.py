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
            # trust overall + 분해 성분(4축·게이트)을 online score로 부착 → Langfuse에서
            # answerability_status별 보류 원인 분포를 집계 가능 (#256). trace_id는 피드백 참조용.
            _score_trust_breakdown(state)
            response.trace_id = current_trace_id() or ""
        return response


# Trust overall 블렌드 성분(4축) + 게이트 3종 — online score 이름 매핑.
# overall은 기존 score 이름(trust_score)을 그대로 보존한다(#137 계약).
_EVIDENCE_AXES = (
    ("trust_score", "overall"),  # 가중 블렌드 (기존 이름 유지)
    ("ev_version_fit", "version_fit_mean"),
    ("ev_coverage", "coverage"),
    ("ev_residual_dup", "residual_duplication"),
    ("ev_authority", "authority_mean"),
)
_GATE_FLAGS = (
    ("gate_conflicting", "conflicting"),
    ("gate_deprecated", "deprecated_only"),
    ("gate_sensitive", "sensitive_block"),
)


def _score_trust_breakdown(state: dict) -> None:
    """Trust overall과 그 분해 성분(4축·게이트)을 현재 trace에 online score로 부착한다 (#256).

    overall 단일값만으로는 '왜 보류됐는지'가 안 보여, 성분을 함께 남겨 Langfuse에서
    answerability_status별 원인 분포(coverage 부족 vs 게이트 vs version_fit)를 집계할 수 있게 한다.
    router 차단(UNANSWERABLE) 경로는 trust가 안 돌아 trust_score=None → 전부 skip.
    getattr 방어 — 부분 생성 TrustScore(overall=...)에도 안전(없는 축은 건너뜀).
    """
    trust = state.get("trust_score")
    if trust is None:  # router 차단(UNANSWERABLE) 등 trust 미실행 경로 → 게이트 포함 전부 skip
        return
    for score_name, attr in _EVIDENCE_AXES:
        value = getattr(trust, attr, None)
        if isinstance(value, int | float) and not isinstance(value, bool):
            score_current_trace(name=score_name, value=float(value))
    gate = state.get("gate_flags")
    if gate is not None:
        for score_name, attr in _GATE_FLAGS:
            score_current_trace(name=score_name, value=float(bool(getattr(gate, attr, False))))


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
