"""Trust Agent 노드 — Evidence Confidence 5축 채점 + 재검색 결정.

규칙/메타데이터 기반(LLM 미사용·결정론). retriever와 answer 사이에 위치한다.
재검색 트리거는 **관련성**(top rerank score < τ) 기준이며, 재시도 시 domain 필터를 해제해
폭을 넓힌다. owner_trust·verification_label은 색인 payload에 소스가 없어 중립 스텁(track-B 의존성).

상태 계약:
    읽기: documents, retry_count, max_retries
    쓰기: trust_score, gate_flags, should_re_retrieve, (재검색 시) retry_count·domain, agent_trace
"""

from __future__ import annotations

from app.agents.answer.answerability import GateFlags
from app.agents.retriever.rerank import recency_factor
from app.agents.state import AgentState, SourceDocument, TrustScore
from app.agents.trust.schema import TrustOutput
from app.config import Settings, get_settings


def _duplication(documents: list[SourceDocument]) -> float:
    """동일 content hash 비율 → 중복도 [0,1] (높을수록 중복 많음)."""
    hashes = [d.hash for d in documents if d.hash]
    if not hashes:
        return 0.0
    return 1.0 - len(set(hashes)) / len(hashes)


def _sensitivity(documents: list[SourceDocument], cap: int) -> float:
    """[MASKED_*] 마커 밀도 → 민감정보 위험 [0,1]."""
    if cap <= 0:
        return 0.0
    masked = sum(d.content_snippet.count("[MASKED_") for d in documents)
    return min(1.0, masked / cap)


def _conflicting(documents: list[SourceDocument], gap: float, floor: float) -> bool:
    """동등 권위 충돌 의심: 서로 다른 page의 top 점수가 (1)둘 다 관련성 floor 이상이고 (2)점수 차 < gap.

    floor 조건이 핵심 — 저관련 결과는 점수가 0 근처로 뭉쳐(차이<gap) 거의 항상 '충돌'로 오탐된다.
    실제 충돌은 "둘 다 충분히 관련 있는데 막상막하"인 경우뿐이므로 floor(=재검색 τ)로 거른다.
    """
    by_page: dict[str, float] = {}
    for d in documents:
        if d.page_id:
            by_page[d.page_id] = max(by_page.get(d.page_id, d.rerank_score), d.rerank_score)
    tops = sorted(by_page.values(), reverse=True)
    return len(tops) >= 2 and tops[1] >= floor and (tops[0] - tops[1]) < gap


def score_trust(documents: list[SourceDocument], settings: Settings) -> TrustOutput:
    """문서 집합을 Evidence Confidence 5축으로 채점한다 (순수 함수)."""
    owner = settings.trust_owner_neutral
    verification = settings.trust_verification_neutral
    if not documents:
        return TrustOutput(
            recency=0.0,
            owner_trust=owner,
            verification_label=verification,
            duplication_conflict=0.0,
            sensitivity_risk=0.0,
            overall=0.0,
        )

    recency = max(recency_factor(d.last_modified, settings.rerank_recency_half_life_days) for d in documents)
    duplication = _duplication(documents)
    sensitivity = _sensitivity(documents, settings.trust_sensitivity_masked_cap)

    weights = (
        settings.trust_w_recency,
        settings.trust_w_owner,
        settings.trust_w_verification,
        settings.trust_w_duplication,
        settings.trust_w_sensitivity,
    )
    wsum = sum(weights) or 1.0
    overall = (
        weights[0] * recency
        + weights[1] * owner
        + weights[2] * verification
        + weights[3] * (1.0 - duplication)
        + weights[4] * (1.0 - sensitivity)
    ) / wsum
    overall = max(0.0, min(1.0, overall))

    return TrustOutput(
        recency=recency,
        owner_trust=owner,
        verification_label=verification,
        duplication_conflict=duplication,
        sensitivity_risk=sensitivity,
        overall=overall,
        gate_conflicting=_conflicting(documents, settings.trust_conflict_score_gap, settings.trust_rerank_floor),
        gate_deprecated_only=False,  # verification 데이터 부재 → 항상 False (track-B)
        gate_sensitive_block=sensitivity >= 1.0,
    )


def should_re_retrieve(documents: list[SourceDocument], settings: Settings, retry_count: int, max_retries: int) -> bool:
    """관련성/커버리지 기준 재검색 여부. retry 한도 초과 시 False(무한루프 방지).

    주의: 리랭커 비활성(rerank_score=0.0) 환경에선 top<τ로 판정돼 1회 재검색이 유발될 수 있다
    (max_retries로 상한). 운영 경로는 리랭커 활성 가정.
    """
    if retry_count >= max_retries:
        return False
    if not documents:
        return True
    if max(d.rerank_score for d in documents) < settings.trust_rerank_floor:
        return True
    return len(documents) < settings.trust_min_docs


async def trust_node(state: AgentState) -> dict:
    """문서를 채점하고 재검색 여부를 결정한다."""
    settings = get_settings()
    documents = state.get("documents", [])
    retry = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.trust_max_retries)

    out = score_trust(documents, settings)
    result: dict = {
        "trust_score": TrustScore(
            recency=out.recency,
            verification_label=out.verification_label,
            owner_trust=out.owner_trust,
            duplication_conflict=out.duplication_conflict,
            sensitivity_risk=out.sensitivity_risk,
            overall=out.overall,
        ),
        "gate_flags": GateFlags(
            conflicting=out.gate_conflicting,
            deprecated_only=out.gate_deprecated_only,
            sensitive_block=out.gate_sensitive_block,
        ),
        "agent_trace": ["trust"],
    }

    re_retrieve = should_re_retrieve(documents, settings, retry, max_retries)
    result["should_re_retrieve"] = re_retrieve
    if re_retrieve:
        result["retry_count"] = retry + 1
        result["domains"] = []  # 재시도: 도메인 가산 해제로 폭 확대 (같은 결과 회피)
        result["domain"] = None  # 하위호환 파생값 동기화
    return result


def trust_decision(state: AgentState) -> str:
    """근거 부족 시 재검색(retriever), 충분하면 answer로 분기."""
    return "retriever" if state.get("should_re_retrieve") else "answer"
