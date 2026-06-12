"""Trust Agent(Evidence Confidence) 채점 결과 스키마 (#108 — 버전 계보 4축 재설계).

규칙 기반 채점기(node.score_survivors)가 산출하는 4축 + overall + 게이트 신호.
`AgentState.trust_score`(state.TrustScore)와 분리해, 계산 결과를 담는 순수 값 객체로 둔다.

**계약 경계**: 이 모듈은 trust_node 내부 전용이다 — 외부(상태/응답)로 나가는 보고 계약은
state.TrustScore가 담당하며, 구 5축 하위호환 별칭(duplication_conflict 등)도 그쪽에 있다.
여기의 구 5축 필드(recency/owner/verification/sensitivity)는 TrustScore 매핑의 원료일 뿐
overall 블렌드에는 들어가지 않는다(설계 5.1: 측정 가능한 축만 블렌드).
"""

from __future__ import annotations

from pydantic import BaseModel

from app.agents.state import RetryAction


class TrustOutput(BaseModel):
    """Evidence Confidence 채점 결과 (전부 [0,1])."""

    # ── 4축 (overall 블렌드 성분, 설계 4장) ──
    version_fit_mean: float  # 생존 문서 version_fit 평균
    coverage: float  # 주제 충분성 (비교 질의=회수율, waiver 시 1.0)
    residual_duplication: float  # collapse 후 잔여 중복 (높을수록 나쁨)
    authority_mean: float  # site 권위 평균
    overall: float  # Final Evidence Score = 가중 블렌드 (answerability 점수 분기 입력)
    waiver_applied: bool = False  # strong-single-topic waiver 발동 여부
    n_good_topics: int = 0  # raw ≥ τ 문서를 가진 distinct 주제 수 (collapse 후)
    # ── 보고 계약 유지 (구 5축 — 블렌드 미포함) ──
    recency: float = 0.0  # 관측값
    owner_trust: float = 1.0  # 중립 상수 (track-B 데이터 부재)
    verification_label: float = 1.0  # 중립 상수
    sensitivity_risk: float = 0.0  # 게이트 전용
    # ── Answerability 게이트 신호 (사다리 소진 후 판정 — 설계 v1.5) ──
    gate_conflicting: bool = False  # 같은 site·같은 product_version의 다른 주제 충돌 의심
    gate_deprecated_only: bool = False  # 생존 문서 전부 EOL
    gate_sensitive_block: bool = False  # 고위험 민감정보 차단


class RetryDecision(BaseModel):
    """재검색 사다리 판정 결과 (설계 6장 — 표 순서 = 우선순위)."""

    action: RetryAction
    version_filter: str = ""  # RETRY_VERSION: product_version 일치 필터 값
    pinned_doc_keys: list[str] = []  # RETRY_VERSION: 대상 doc_key 고정
    excluded_doc_keys: list[str] = []  # EXPAND_TOPICS: 확보 주제 제외
