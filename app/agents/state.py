"""
OnRamp AgentState 정의.

모든 Agent가 공유하는 상태(State) 스키마.
LangGraph는 TypedDict 기반으로 동작하며, 각 Agent는 자신이 담당하는 필드만 쓰고 반환한다.

읽기/쓰기 매핑 (실행 순서):
    Router    — 읽기: query                  → 쓰기: use_case, domain, refined_query
    Retriever — 읽기: refined_query          → 쓰기: documents
    Trust     — 읽기: documents              → 쓰기: trust_score, should_re_retrieve  (Sprint 3 P1)
                · 문서를 Evidence Confidence 5축으로 채점한다.
                · 근거가 부족하면 should_re_retrieve로 재검색(retriever) 루프를 돈다.
    Answer    — 읽기: documents, trust_score → 쓰기: answer, sources,
                                                     answerability_status, answerability_reason
                · 재검색을 거쳐도 점수가 낮으면 Answerability Status로 처리 방식을 정한다.
                  (답변 생성 / 부분 답변 / 보류 / 충돌 안내)
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
    """Confluence 문서 5도메인 분류.

    값은 인덱싱 분류기(app/rag/classifier.py, 정의: docs/Hyeonmoon/05_classifier.md)가
    Qdrant payload `domain`에 쓰는 **영문 키**와 일치한다. Router(질문 분류)와
    Auto-Classifier(문서 태깅)가 동일 키를 써야 Retriever의 도메인 필터가 동작한다.
    """

    INCIDENT = "incident"  # 장애 대응, 원인 분석, 재발 방지
    MANUAL = "manual"  # 설치, 설정, 운영 절차, How-to (기본값)
    API_REFERENCE = "api_reference"  # API 명세, 파라미터, 명령어 레퍼런스
    MEETING_NOTE = "meeting_note"  # 회의록, 의사결정 기록
    PLANNING = "planning"  # 설계 문서, 아키텍처, 기획서, RFC/PRD


class UseCase(StrEnum):
    """사용자 의도 분류 (Router). ASSET은 없음 — 자산화는 /v1/asset API로 구분한다.

    UNANSWERABLE은 '검색 범위를 벗어난 질문'을 retrieve 전에 차단하는 용도다.
    검색 후 근거가 부족한 경우는 UseCase가 아니라 AnswerabilityStatus로 다룬다.
    """

    SEARCH = "검색"
    UNANSWERABLE = "답변불가"


class AnswerabilityStatus(StrEnum):
    """답변 가능성 상태 (Answer Agent의 최종 처리 방식). Sprint 3 P1.

    Trust(Evidence Confidence)가 재검색 루프로도 근거를 충분히 채우지 못하면,
    Final Evidence Score를 기준으로 이 상태를 정해 답변 생성/보류/안내를 분기한다.
    """

    ANSWERABLE = "answerable"  # 근거 충분 → 일반 답변 생성
    PARTIALLY_ANSWERABLE = "partially_answerable"  # 부분 근거 → 한계 명시 답변
    NOT_ENOUGH_EVIDENCE = "not_enough_evidence"  # 근거 부족 → 답변 보류 + 추가 검색 안내
    CONFLICTING_EVIDENCE = "conflicting_evidence"  # 동등 권위 문서 충돌 → 버전 확인 요청
    OUTDATED_EVIDENCE = "outdated_evidence"  # 최신 문서 부재 → 제한 답변


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
    domain: Domain | None  # None이면 Retriever가 도메인 필터 없이 검색
    refined_query: str  # 검색용 정제 쿼리

    # ── Retriever Agent 출력 ──
    documents: list[SourceDocument]

    # ── Trust Agent 출력 (Evidence Confidence, Sprint 3 P1 — 타입만 정의) ──
    #    문서 기반 5축 채점 → 근거 부족 시 should_re_retrieve로 retriever 재검색 루프
    trust_score: TrustScore
    should_re_retrieve: bool
    retry_count: int
    max_retries: int

    # ── Answer Agent 출력 ──
    answer: FiveElements
    sources: list[SourceDocument]  # 답변에 인용된 문서
    answerability_status: AnswerabilityStatus
    answerability_reason: str  # 상태별 사용자 안내 메시지

    # ── 메타 (디버깅·로깅) ──
    error: str
    agent_trace: Annotated[list[str], add]  # add reducer → 각 Agent가 반환한 리스트가 자동 누적
