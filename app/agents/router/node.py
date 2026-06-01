"""Router Agent 노드 — 5도메인 분류 + 쿼리 정제.

LLM 1회 호출로 use_case·domain·refined_query를 동시에 산출한다.
async 노드이므로 그래프는 ainvoke로 실행한다 (retriever와 동일).
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from app.agents.router.prompts import ROUTER_SYSTEM_PROMPT
from app.agents.router.schema import RouterOutput
from app.agents.state import AgentState, Domain, UseCase
from app.services.llm_selector import call_llm

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.5  # 미만이면 domain을 fallback으로
_FALLBACK_DOMAIN = Domain.OPS_MANUAL


def _fallback(query: str, error: str = "") -> dict:
    """LLM 실패·파싱 실패 시 기본값 (검색은 되게 SEARCH로)."""
    result: dict = {
        "use_case": UseCase.SEARCH,
        "domain": _FALLBACK_DOMAIN,
        "refined_query": query,
        "agent_trace": ["router"],
    }
    if error:
        result["error"] = error
    return result


async def route_node(state: AgentState) -> dict:
    """사용자 질문을 5도메인으로 분류하고 검색 쿼리를 정제한다."""
    query = state["query"]
    model = state.get("model", "")

    try:
        raw = await call_llm(ROUTER_SYSTEM_PROMPT, query, model=model, json_mode=True)
    except Exception as exc:  # LLM 호출 실패 → error 기록 후 fallback
        logger.warning("Router LLM 호출 실패 — 기본값 fallback", exc_info=True)
        return _fallback(query, error=str(exc))

    try:
        output = RouterOutput.model_validate_json(raw)
    except ValidationError:  # JSON/스키마 파싱 실패 → 검색 fallback
        logger.warning("Router 응답 파싱 실패 — 기본값 fallback", exc_info=True)
        return _fallback(query)

    # confidence 낮으면 도메인만 fallback (검색 자체는 진행)
    domain = output.domain if output.confidence >= _CONFIDENCE_THRESHOLD else _FALLBACK_DOMAIN
    # UNANSWERABLE이면 LLM 출력과 무관하게 refined_query를 비운다 (노드에서 계약 보장)
    refined_query = "" if output.use_case == UseCase.UNANSWERABLE else output.refined_query
    return {
        "use_case": output.use_case,
        "domain": domain,
        "refined_query": refined_query,
        "agent_trace": ["router"],
    }
