"""Router Agent 노드 — 5도메인 분류 + 쿼리 정제.

LLM 1회 호출로 use_case·domains·refined_query를 동시에 산출한다.
async 노드이므로 그래프는 ainvoke로 실행한다 (retriever와 동일).

운영(route_node)과 평가(예측 캐시)가 **같은 LLM 호출·파싱·fallback·신뢰도 게이팅**을
쓰도록 핵심 로직을 ``classify_query``로 분리한다. route_node는 그 결과를 AgentState
부분집합 dict로 매핑만 하므로 운영 동작은 분리 전과 동일하다(회귀 테스트로 고정).
평가는 ``classify_query``를 직접 호출해 confidence/parse_ok/fallback 같은 진단값을
얻는다 — 평가 코드에서 LLM 호출·파싱 로직을 복제하지 않는다.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.agents.router.prompts import ROUTER_SYSTEM_PROMPT
from app.agents.router.schema import RouterOutput
from app.agents.state import AgentState, Domain, UseCase
from app.services.llm_selector import call_llm

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.5  # 이 미만이면 도메인 미신뢰 → 빈 리스트(가산 미적용)
_UNANSWERABLE_REASON = "사내 지식 범위를 벗어난 질문입니다."
# target_versions 방어 필터 (#108) — LLM이 "latest" 류 비숫자 토큰을 넣으면 제거.
# '최신' 질의는 match가 아니라 currency 모드가 정답이라 target으로 인정하지 않는다.
_VERSION_TOKEN_RE = re.compile(r"^v?\d+(\.\d+)*$")


def _valid_versions(values: list[str]) -> list[str]:
    return [v.strip() for v in values if _VERSION_TOKEN_RE.match(v.strip())]


@dataclass(frozen=True)
class RouterDiagnostics:
    """라우터 1회 분류의 진단 결과 (운영 매핑 + 평가 캐시 공용).

    operational 필드(use_case·domains·refined_query)와 진단 필드(raw_domains·
    confidence·parse_ok·fallback_reason)를 함께 담아, route_node는 전자를, 평가
    캐시는 후자를 사용한다. AgentState를 진단값으로 오염시키지 않기 위한 분리다.
    """

    use_case: UseCase
    domains: list[Domain]  # confidence 게이팅 후 — 운영/검색이 실제 쓰는 예측값
    raw_domains: list[Domain]  # 게이팅 전(파싱 직후) — 저신뢰로 비워졌는지 구분용
    confidence: float | None  # 실패(llm_error/parse_error)면 None — ECE 왜곡 방지
    parse_ok: bool  # RouterOutput 파싱 성공 여부
    fallback_reason: str | None  # None | "llm_error" | "parse_error"
    refined_query: str
    error: str | None = None  # llm_error 시 예외 메시지 (route_node의 error 키 패리티)
    target_versions: list[str] = field(default_factory=list)  # 질의 명시 구체 버전 (#108)


async def classify_query(query: str, model: str = "") -> RouterDiagnostics:
    """질문을 1회 LLM 호출로 분류한다 — 운영·평가 공용 핵심 로직.

    실패 처리(운영 route_node와 동일 의미):
      · LLM 호출 실패 → fallback_reason="llm_error", confidence=None
      · 파싱/스키마 실패 → fallback_reason="parse_error", confidence=None
    정상: confidence가 임계값 미만이면 domains를 비운다(raw_domains에는 원본 보존).
    """
    try:
        raw = await call_llm(ROUTER_SYSTEM_PROMPT, query, model=model, json_mode=True)
    except Exception as exc:  # LLM 호출 실패 → error 기록 후 fallback
        logger.warning("Router LLM 호출 실패 — 기본값 fallback", exc_info=True)
        return RouterDiagnostics(
            use_case=UseCase.SEARCH,
            domains=[],
            raw_domains=[],
            confidence=None,
            parse_ok=False,
            fallback_reason="llm_error",
            refined_query=query,
            error=str(exc),
        )

    try:
        output = RouterOutput.model_validate_json(raw)
    except ValidationError:  # JSON/스키마 파싱 실패 → 검색 fallback
        logger.warning("Router 응답 파싱 실패 — 기본값 fallback", exc_info=True)
        return RouterDiagnostics(
            use_case=UseCase.SEARCH,
            domains=[],
            raw_domains=[],
            confidence=None,
            parse_ok=False,
            fallback_reason="parse_error",
            refined_query=query,
            error=None,
        )

    # confidence가 낮으면 도메인을 신뢰하지 않고 빈 리스트로 둔다 (가산 미적용).
    raw_domains = list(output.domains)
    gated = raw_domains if output.confidence >= _CONFIDENCE_THRESHOLD else []
    return RouterDiagnostics(
        use_case=output.use_case,
        domains=gated,
        raw_domains=raw_domains,
        confidence=output.confidence,
        parse_ok=True,
        fallback_reason=None,
        refined_query=output.refined_query,
        error=None,
        target_versions=_valid_versions(output.target_versions),
    )


def _fallback(query: str, error: str = "") -> dict:
    """LLM/파싱 실패 시 기본 상태. 검색은 진행하되 도메인은 없음(가산 미적용)."""
    result: dict = {
        "use_case": UseCase.SEARCH,
        "domains": [],
        "domain": None,  # 하위호환: domains[0] 파생 (빈 리스트 → None)
        "refined_query": query,
        "target_versions": [],
        "agent_trace": ["router"],
    }
    if error:
        result["error"] = error
    return result


async def route_node(state: AgentState) -> dict:
    """사용자 질문을 5도메인으로 분류하고 검색 쿼리를 정제한다.

    ``classify_query`` 결과를 AgentState 부분집합으로 매핑만 한다(분리 전과 동일 동작).
    """
    query = state["query"]
    model = state.get("model", "")

    diag = await classify_query(query, model=model)
    if diag.fallback_reason == "llm_error":
        return _fallback(query, error=diag.error or "")
    if diag.fallback_reason == "parse_error":
        return _fallback(query)

    domains = diag.domains
    result: dict = {
        "use_case": diag.use_case,
        "domains": domains,
        "domain": domains[0] if domains else None,  # 하위호환: 항상 domains[0] 파생(불일치 금지)
        "refined_query": diag.refined_query,
        "target_versions": diag.target_versions,  # 1차 추출값 — 재작성 재검색에도 고정(모드 보존)
        "agent_trace": ["router"],
    }
    # UNANSWERABLE이면 LLM 출력과 무관하게 refined_query를 비우고 안내 사유를 채운다 (노드 계약 보장)
    if diag.use_case == UseCase.UNANSWERABLE:
        result["refined_query"] = ""
        result["target_versions"] = []
        result["answerability_reason"] = _UNANSWERABLE_REASON
    return result
