"""Router Agent 노드 — 5도메인 분류 + 쿼리 정제.

LLM 1회 호출로 use_case·domain·refined_query를 동시에 산출한다.
async 노드이므로 그래프는 ainvoke로 실행한다 (retriever와 동일).
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from app.agents.router.prompts import ROUTER_SYSTEM_PROMPT
from app.agents.router.schema import RouterOutput
from app.agents.state import AgentState, UseCase
from app.services.llm_selector import call_llm

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.5  # 이 미만이면 도메인 미신뢰 → None
_UNANSWERABLE_REASON = "사내 지식 범위를 벗어난 질문입니다."


def _fallback(query: str, error: str = "") -> dict:
    """LLM/파싱 실패 시 기본 상태. 검색은 진행하되 도메인은 없음(가산 미적용)."""
    result: dict = {
        "use_case": UseCase.SEARCH,
        "domains": [],
        "domain": None,  # 하위호환: domains[0] 파생 (빈 리스트 → None)
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

    # confidence가 낮으면 도메인을 신뢰하지 않고 빈 리스트로 둔다 (가산 미적용).
    domains = list(output.domains) if output.confidence >= _CONFIDENCE_THRESHOLD else []
    result: dict = {
        "use_case": output.use_case,
        "domains": domains,
        "domain": domains[0] if domains else None,  # 하위호환: 항상 domains[0] 파생(불일치 금지)
        "refined_query": output.refined_query,
        "agent_trace": ["router"],
    }
    # UNANSWERABLE이면 LLM 출력과 무관하게 refined_query를 비우고 안내 사유를 채운다 (노드 계약 보장)
    if output.use_case == UseCase.UNANSWERABLE:
        result["refined_query"] = ""
        result["answerability_reason"] = _UNANSWERABLE_REASON
    return result
