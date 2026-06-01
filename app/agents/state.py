"""
OnRamp AgentState 정의.

모든 Agent가 공유하는 상태(State) 스키마.
LangGraph는 TypedDict 기반으로 동작하며, 각 Agent는 자신이 담당하는 필드만 쓰고 반환한다.

읽기/쓰기 매핑:
    Router    — 읽기: query           → 쓰기: use_case, domain, refined_query
    Retriever — 읽기: refined_query   → 쓰기: documents
    Answer    — 읽기: documents       → 쓰기: answer, sources, is_answerable, unanswerable_reason
    Trust     — 읽기: answer          → 쓰기: trust_score, should_re_retrieve  (Sprint 3 P1)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from operator import add
from typing import Annotated, TypedDict

# ---------------------------------------------------------------------------
# 도메인 & 유스케이스 분류
# ---------------------------------------------------------------------------


class Domain(StrEnum):
    """Confluence 문서 5도메인 분류. Router와 Auto-Classifier가 동일 Enum을 공유한다."""

    INCIDENT = "장애대응"
    OPS_MANUAL = "운영매뉴얼"
    API_SPEC = "API명세"
    MEETING_NOTES = "회의록"
    PLANNING = "기획서"


class UseCase(StrEnum):
    """사용자 의도 분류. ASSET은 없음 — 자산화는 /v1/asset API로 구분한다."""

    SEARCH = "검색"
    UNANSWERABLE = "답변불가"


# ---------------------------------------------------------------------------
# 중첩 타입 (dataclass)
# ---------------------------------------------------------------------------


@dataclass
class SourceDocument:
    """검색된 출처 문서 한 건."""

    title: str = ""
    url: str = ""
    space_key: str = ""
    content_snippet: str = ""
    score: float = 0.0  # 벡터 검색 유사도
    rerank_score: float = 0.0  # Cross-Encoder 리랭킹 점수


@dataclass
class FiveElements:
    """5요소 구조화 답변. Answer Agent와 Asset Service가 동일 스키마를 사용한다."""

    situation: str = ""  # 현재 상황
    cause: str = ""  # 원인
    evidence: str = ""  # 근거
    solution: str = ""  # 해결
    infra_context: str = ""  # 인프라 맥락


@dataclass
class TrustScore:
    """Evidence Confidence 5축 신뢰도 평가 결과. Sprint 3 P1에서 Trust Agent가 사용한다.

    5축 점수에 Intent-Document Fit / Document Base Score를 더해
    Final Evidence Score(``overall``)를 산출한다.
    """

    recency: float = 0.0  # 최신성
    verification_label: float = 0.0  # 검증 라벨 (검증됨/초안 등)
    owner_trust: float = 0.0  # 소유자 신뢰도
    duplication_conflict: float = 0.0  # 중복도/충돌
    sensitivity_risk: float = 0.0  # 민감정보 위험
    overall: float = 0.0  # Final Evidence Score (종합 점수)


# ---------------------------------------------------------------------------
# AgentState (LangGraph 공유 상태)
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """
    LangGraph StateGraph의 공유 상태.

    total=False → 모든 필드가 optional.
    Agent 노드는 변경된 필드만 dict로 반환하면 LangGraph가 자동 머지한다.
    """

    # ── 사용자 입력 ──
    query: str
    model: str  # LLM 모델명 (빈값이면 config 기본값)

    # ── Router Agent 출력 ──
    use_case: UseCase
    domain: Domain
    refined_query: str  # 검색용 정제 쿼리

    # ── Retriever Agent 출력 ──
    documents: list[SourceDocument]

    # ── Answer Agent 출력 ──
    answer: FiveElements
    sources: list[SourceDocument]  # 답변에 인용된 문서
    is_answerable: bool
    unanswerable_reason: str

    # ── 메타 (디버깅·로깅) ──
    error: str
    agent_trace: Annotated[list[str], add]  # add reducer → 각 Agent가 반환한 리스트가 자동 누적

    # ── Sprint 3 P1 (타입만 정의, Trust 노드 미연결) ──
    trust_score: TrustScore
    should_re_retrieve: bool
    retry_count: int
    max_retries: int
