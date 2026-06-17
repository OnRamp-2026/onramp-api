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


class RetryAction(StrEnum):
    """Trust 재검색 사다리의 액션 (#108, 설계 6장).

    실패 원인에 전략을 맞춘다 — 같은 검색의 반복이 아니라 전략 변형.
    판정 우선순위는 trust_node.decide_retry_action의 선언 순서(표 순서 = 우선순위).
    """

    REWRITE_QUERY = "rewrite_query"  # 관련 근거 전무(n_good_topics=0) → LLM 쿼리 재작성
    RETRY_VERSION = "retry_version"  # 미회수/옛 버전 → doc_key 고정 + product_version 필터
    EXPAND_TOPICS = "expand_topics"  # 주제 부족 → 도메인 해제 + top_k 확대 + doc_key 제외
    PROCEED = "proceed"  # 재검색 불필요 (또는 한도 소진 강제 진행)


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
    # 점수 분리 (#103, 설계 7.3): 정렬은 블렌드(rerank_score), 진단은 원점수(raw_rerank_score)
    rerank_score: float = 0.0  # ranking 점수 — 부스트(최신성·도메인·버전·권위) 합산, 정렬 전용
    raw_rerank_score: float = 0.0  # Cross-Encoder 원점수 [0,1] — τ 진단 전용 (리랭커 비활성 시 0.0)
    # Trust Agent(Evidence Confidence) 채점 입력 (응답에는 노출 안 함)
    page_id: str = ""  # 중복/충돌 판단 (서로 다른 page 간)
    last_modified: str = ""  # 최신성(recency)
    hash: str = ""  # 중복 content 탐지
    # 버전 계보 메타 (#94 payload → #103 전달)
    chunk_id: str = ""  # 재검색 병합 dedupe 키
    site: str = ""
    product_version: str = ""
    doc_key: str = ""  # 버전 형제 묶음 키 (빈 값 = 계보 없음)
    is_eol: bool = False
    # Trust per-doc 채점 (#108 — trust_node가 기입, Answer 인용 우선순위에 사용)
    version_fit: float = 0.0  # 버전 적합성 [0,1] (currency/match 모드)
    version_fit_mode: str = ""  # "currency" | "match" — 관측·디버깅용
    raw_currency: float = 0.0  # 디버그 + collapse 타이브레이커
    per_doc_evidence: float = 0.0  # w_version·fit + w_authority·tier — 인용 우선순위


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
    """Evidence Confidence 신뢰도 평가 결과 (#108 — 버전 계보 4축 재설계).

    채점 차원은 4축(version_fit·coverage·residual_duplication·authority)이며
    overall은 그 가중 블렌드다(설계 5.1). 기존 5필드는 보고 계약 유지를 위해 남긴다 —
    recency/sensitivity_risk는 관측값(블렌드 미포함), owner/verification은 중립 상수.
    """

    # ── 보고 계약 유지 필드 (구 5축 — overall 블렌드에는 미포함) ──
    recency: float = 0.0  # 최신성 관측값 (crawled 코퍼스에선 업로드 시각이라 참고용)
    # 기본값 1.0 — trust/schema.py의 중립 상수와 일치. 부분 생성(TrustScore(overall=...))
    # 경로에서 "최저 신뢰(0.0)"로 잘못 보고되지 않게 한다.
    verification_label: float = 1.0  # 중립 상수 (track-B 데이터 부재)
    owner_trust: float = 1.0  # 중립 상수 (track-B 데이터 부재)
    duplication_conflict: float = 0.0  # = residual_duplication (하위호환 별칭)
    sensitivity_risk: float = 0.0  # 민감정보 위험 — 게이트 전용 (블렌드 제외)
    overall: float = 0.0  # Final Evidence Score (4축 가중 블렌드)
    # ── 4축 재설계 (#108, 설계 4장) ──
    version_fit_mean: float = 0.0  # 생존 문서 per_doc_evidence 평균의 버전 성분
    coverage: float = 0.0  # 주제 충분성 (비교 질의는 회수율)
    residual_duplication: float = 0.0  # collapse 후 잔여 중복
    authority_mean: float = 0.0  # site 권위 평균
    waiver_applied: bool = False  # strong-single-topic waiver 발동 여부


@dataclass
class GateFlags:
    """Answerability 게이트 신호 (Trust가 채움). P0에선 미사용.

    state.py에 두어 AgentState 어노테이션의 런타임 평가(LangGraph get_type_hints)와
    answerability↔state 순환 import를 모두 피한다. answerability는 여기서 re-export.
    """

    conflicting: bool = False  # 동등 권위 문서 간 내용 충돌
    deprecated_only: bool = False  # deprecated/archived 문서만 검색됨
    sensitive_block: bool = False  # 고위험 민감정보 차단


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
    domains: list[Domain]  # 질의 도메인(순서=우선순위, 최대 2). 빈 리스트면 가산 미적용
    domain: Domain | None  # 하위호환 파생값 = domains[0] (없으면 None). 별도 판단 아님
    refined_query: str  # 검색용 정제 쿼리
    # 질의가 명시한 구체 target 버전 (#108 — Router가 추출, '최신/latest' 표현은 추출 금지).
    # 복수면 버전 비교 질의. 재작성 재검색 시에도 1차 추출값 고정(모드 플립 방지, 설계 v1.5).
    target_versions: list[str]

    # ── Retriever Agent 출력 ──
    documents: list[SourceDocument]

    # ── Trust Agent 출력 (Evidence Confidence, #108 재설계) ──
    #    병합 → per-doc 채점 → collapse → coverage → overall → 재검색 사다리 → (소진 후) 게이트
    trust_score: TrustScore
    gate_flags: GateFlags  # Answerability 게이트 신호 (answer 노드가 소비)
    should_re_retrieve: bool  # trust_decision 분기용 (retry_action != PROCEED)
    retry_count: int
    max_retries: int
    # 재검색 사다리 상태 (#108, 설계 6장 — trust가 쓰고 retriever가 소비)
    retry_action: RetryAction  # 이번 재검색의 전략 (PROCEED면 재검색 없음)
    excluded_doc_keys: list[str]  # EXPAND_TOPICS: 확보한 doc_key 제외 (새 주제 발견 목적)
    pinned_doc_keys: list[str]  # RETRY_VERSION: 대상 doc_key 고정
    version_filter: str  # RETRY_VERSION: product_version 일치 필터 (payload 표기값)
    first_pass_documents: list[SourceDocument]  # 병합 규칙(v1.4): 최종 = 1차 생존 ∪ 재검색 결과
    missing_versions: list[str]  # 비교 질의에서 회수 못 한 target 버전 (PARTIALLY 사유용)

    # ── Answer Agent 출력 ──
    answer: FiveElements  # structured 포맷일 때 채움 (freeform이면 빈값)
    answer_text: str  # freeform 포맷일 때 채움 (structured면 "")
    answer_format: str  # "structured" | "freeform" — 라우터 도메인 기준(#191)
    sources: list[SourceDocument]  # 답변에 인용된 문서
    answerability_status: AnswerabilityStatus
    answerability_reason: str  # 상태별 사용자 안내 메시지

    # ── 메타 (디버깅·로깅) ──
    error: str
    agent_trace: Annotated[list[str], add]  # add reducer → 각 Agent가 반환한 리스트가 자동 누적
