"""Trust Agent(Evidence Confidence) 채점 결과 스키마.

규칙 기반 채점기(node.score_trust)가 산출하는 5축 + overall + gate 신호.
`AgentState.trust_score`(state.TrustScore)와 분리해, 계산 결과를 담는 순수 값 객체로 둔다.
"""

from __future__ import annotations

from pydantic import BaseModel


class TrustOutput(BaseModel):
    """Evidence Confidence 5축 채점 결과 (전부 [0,1])."""

    recency: float  # 최신성 (높을수록 최신)
    owner_trust: float  # 소유자 신뢰 (데이터 부재 → 중립 스텁, track-B)
    verification_label: float  # 검증 라벨 (데이터 부재 → 중립 스텁, track-B)
    duplication_conflict: float  # 중복도 (높을수록 중복 많음 = 나쁨)
    sensitivity_risk: float  # 민감정보 위험 (높을수록 위험 = 나쁨)
    overall: float  # 종합 Evidence Score (good-direction 가중 블렌드)
    # Answerability 게이트 신호 (answer 노드가 GateFlags로 전달)
    gate_conflicting: bool = False  # 동등 권위 문서 충돌 의심
    gate_deprecated_only: bool = False  # deprecated/archived만 (verification 데이터 필요 → 현재 항상 False)
    gate_sensitive_block: bool = False  # 고위험 민감정보 차단
