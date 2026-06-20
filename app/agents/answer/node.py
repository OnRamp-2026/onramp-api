"""Answer Agent 노드 — 조건부 포맷(구조화/freeform) 답변 생성 + Answerability Status 판정.

포맷은 **사용자 의도(라우터 domains)** 로만 결정한다(#191): incident면 5요소 구조화, 그 외는 freeform 산문.
근거 품질(answerability)·grounding·sources·보류는 **두 포맷 공통**이며 포맷과 무관하게 동일 로직을 재사용한다.

P0: 결정론 floor(문서 0건·LLM 실패·파싱 실패) + LLM 자기판정 + 인용 guard 하이브리드.
P1: Trust(Evidence Confidence)가 evidence_score·gate를 넘기면 decide_answerability가 점수/게이트로 분기.
async 노드이므로 그래프는 ainvoke로 실행한다.
"""

from __future__ import annotations

import logging

from app.agents.answer.answerability import (
    NO_SOURCE_REASON,
    decide_answerability,
    reason_for,
)
from app.agents.answer.formatter import format_answer, format_freeform
from app.agents.answer.prompts import ANSWER_SYSTEM_PROMPT, FREEFORM_SYSTEM_PROMPT
from app.agents.format_policy import decide_answer_format
from app.agents.state import AgentState, AnswerabilityStatus, FiveElements, SourceDocument
from app.config import get_settings
from app.services.llm_selector import call_llm

logger = logging.getLogger(__name__)

# 보류 상태 — 본문(5요소/answer_text)을 비우고 안내 메시지만 제공
_HOLD_STATUSES = {
    AnswerabilityStatus.NOT_ENOUGH_EVIDENCE,
    AnswerabilityStatus.CONFLICTING_EVIDENCE,
    AnswerabilityStatus.OUTDATED_EVIDENCE,
}


def _result(
    *,
    answer_format: str,
    answer: FiveElements,
    answer_text: str,
    sources: list[SourceDocument],
    status: AnswerabilityStatus,
    reason: str,
    error: str = "",
) -> dict:
    out: dict = {
        "answer": answer,
        "answer_text": answer_text,
        "answer_format": answer_format,
        "sources": sources,
        "answerability_status": status,
        "answerability_reason": reason,
        "agent_trace": ["answer"],
    }
    if error:
        out["error"] = error
    return out


def _window_parent(parent: str, child: str, window_chars: int) -> str:
    """parent 본문을 matched child 주변 ~window_chars 범위로 좁힌다 (#212 step7 — parent 비용 절감).

    parent가 이미 예산 내면 그대로 반환. child 위치는 snippet 앞부분(probe)으로 찾고, 못 찾으면
    앞에서 자른다. **문자 단위로 슬라이스**해 서식(코드/표)을 최대한 보존한다.
    """
    if window_chars <= 0 or len(parent) <= window_chars:
        return parent
    probe = child.strip()[: min(80, window_chars)]  # probe도 예산 내로 (작은 window 대비)
    pos = parent.find(probe) if probe else -1
    if pos < 0:
        return parent[:window_chars].rstrip()  # child 못 찾으면 앞부분
    half = max(0, (window_chars - len(probe)) // 2)
    start = max(0, pos - half)
    end = min(len(parent), start + window_chars)  # 예산 상한 고정 (len(segment) <= window_chars)
    return parent[start:end].strip()


def _select_contexts(
    documents: list[SourceDocument], parent_contexts: dict[str, str] | None
) -> list[tuple[int, str, str]]:
    """각 문서가 LLM에 넣을 (원본 index, title, content)를 고른다 — 컨텍스트 선택의 단일 소스.

    parent_contexts가 있으면(#212 parent-expanded) parent 본문을 parent_id 기준 **한 번만**
    쓰고(여러 child가 같은 parent면 중복 제거), 비면 child snippet(=child-only baseline).
    `parent_context_window_chars>0`면 parent를 matched child 주변 window로 좁힌다(step7 비용 절감).
    인덱스는 **원본 documents 인덱스**를 유지한다 — formatter가 LLM 인용 [i]를 documents[i]로
    역매핑하므로(재번호 금지). dedupe된 child는 건너뛰되 인덱스는 보존.
    """
    window = get_settings().parent_context_window_chars
    blocks: list[tuple[int, str, str]] = []
    seen_parents: set[str] = set()
    for i, doc in enumerate(documents):
        pid = doc.parent_id
        if parent_contexts and pid and pid in parent_contexts:
            if pid in seen_parents:
                continue  # 같은 parent는 한 번만 (그 parent는 먼저 나온 child 인덱스로 인용됨)
            seen_parents.add(pid)
            content = parent_contexts[pid]
            if window > 0:  # matched child(=이 doc) 주변으로 trimming
                content = _window_parent(content, doc.content_snippet, window)
        else:
            content = doc.content_snippet  # parent 없는 문서는 child snippet fallback
        blocks.append((i, doc.title, content))
    return blocks


def _build_context(documents: list[SourceDocument], parent_contexts: dict[str, str] | None = None) -> str:
    """LLM 컨텍스트 문자열. 선택 규칙은 _select_contexts와 공유(평가 retrieved_contexts와 일치 보장)."""
    return "\n\n".join(
        f"[{i}] title: {title}\ncontent: {content}"
        for i, title, content in _select_contexts(documents, parent_contexts)
    )


def context_contents(documents: list[SourceDocument], parent_contexts: dict[str, str] | None = None) -> list[str]:
    """LLM이 **실제로 본** context 본문 리스트 (평가 retrieved_contexts와 동일 소스, #212).

    parent-expanded면 parent 본문, 아니면 child snippet. 빈 본문은 제외(RAGAS 채점에 무의미).
    이걸로 RAGAS를 채점해야 parent 모드 faithfulness가 'LLM이 본 문맥' 기준으로 공정해진다.
    """
    return [content for _, _, content in _select_contexts(documents, parent_contexts) if content]


async def _fetch_parent_contexts(documents: list[SourceDocument], tenant_id: str | None = None) -> dict[str, str]:
    """parent_context_enabled일 때 검색 문서의 parent 본문을 Postgres에서 조회(parent_id dedupe)."""
    settings = get_settings()
    if not settings.parent_context_enabled:
        return {}
    parent_ids = [d.parent_id for d in documents if d.parent_id]
    if not parent_ids:
        return {}
    from app.db.postgres import session_scope
    from app.services import rag_index_repository as repo

    try:
        async with session_scope() as db:
            return await repo.get_parent_contexts(
                db,
                tenant_id=tenant_id or settings.auth_default_tenant,
                parent_ids=parent_ids,
            )
    except Exception:  # 조회 실패 → child-only로 graceful degrade (답변은 계속 나옴)
        logger.warning("parent context 조회 실패 — child-only로 진행", exc_info=True)
        return {}


async def answer_node(state: AgentState) -> dict:
    """문서 근거로 답변을 생성하고 Answerability Status를 판정한다 (포맷은 라우터 domains 기준)."""
    documents = state.get("documents", [])
    query = state.get("query") or state.get("refined_query", "")
    # 포맷은 라우터가 의도-time에 박은 값을 우선 사용한다 (Trust가 domains를 변형해도 불변, #191).
    # 라우터 미경유 경로(직접 호출 등)를 위해 domains 기반 계산을 fallback으로 둔다.
    answer_format = state.get("answer_format") or decide_answer_format(
        state.get("domains") or [], get_settings().structured_answer_domains
    )
    is_structured = answer_format == "structured"

    # 결정론 floor: 문서 0건 → 보류 (LLM 호출 안 함)
    if not documents:
        status = decide_answerability(documents)
        return _result(
            answer_format=answer_format,
            answer=FiveElements(),
            answer_text="",
            sources=[],
            status=status,
            reason=reason_for(status),
        )

    parent_contexts = await _fetch_parent_contexts(
        documents,
        state.get("tenant_id"),
    )  # #212: parent-expanded면 채워짐, 아니면 {}
    system_prompt = ANSWER_SYSTEM_PROMPT if is_structured else FREEFORM_SYSTEM_PROMPT
    user_prompt = f"문서 컨텍스트:\n{_build_context(documents, parent_contexts)}\n\n질문: {query}"
    try:
        raw = await call_llm(system_prompt, user_prompt, model=state.get("model", ""), json_mode=True)
    except Exception as exc:  # LLM 호출 실패 → 보류
        logger.warning("Answer LLM 호출 실패 — NOT_ENOUGH fallback", exc_info=True)
        status = AnswerabilityStatus.NOT_ENOUGH_EVIDENCE
        return _result(
            answer_format=answer_format,
            answer=FiveElements(),
            answer_text="",
            sources=[],
            status=status,
            reason=reason_for(status),
            error=str(exc),
        )

    # 포맷별 파싱 — 본문만 다르고(5요소 vs answer_text) sources·llm_status·parse_ok 계약은 동일
    five = FiveElements()
    answer_text = ""
    if is_structured:
        five, sources, llm_status, parse_ok = format_answer(raw, documents)
    else:
        answer_text, sources, llm_status, parse_ok = format_freeform(raw, documents)
    if not parse_ok:  # JSON/스키마 파싱 실패 → 보류
        logger.warning("Answer 응답 파싱 실패 — NOT_ENOUGH fallback (format=%s)", answer_format)
        return _result(
            answer_format=answer_format,
            answer=FiveElements(),
            answer_text="",
            sources=[],
            status=AnswerabilityStatus.NOT_ENOUGH_EVIDENCE,
            reason="답변 생성 실패 (파싱 오류)",
        )

    # 인용 우선순위 (#108): Trust per-doc evidence가 높은 근거를 앞세운다 (안정 정렬 — 동률은 원 순서)
    sources = sorted(sources, key=lambda d: d.per_doc_evidence, reverse=True)

    # Trust 게이트(충돌/만료/민감차단) 우선 → evidence_score → LLM 자기판정 순. (포맷 무관 — 공통)
    gate = state.get("gate_flags")
    trust = state.get("trust_score")
    status = decide_answerability(
        documents,
        evidence_score=(trust.overall if trust else None),
        gate=gate,
        llm_status=llm_status,
    )

    # 인용 guard: ANSWERABLE인데 인용 출처 0건 → PARTIALLY로 강등
    if status == AnswerabilityStatus.ANSWERABLE and not sources:
        status, reason = AnswerabilityStatus.PARTIALLY_ANSWERABLE, NO_SOURCE_REASON
    else:
        reason = reason_for(status)
    if (
        gate
        and gate.deprecated_warning
        and status
        in {
            AnswerabilityStatus.ANSWERABLE,
            AnswerabilityStatus.PARTIALLY_ANSWERABLE,
        }
    ):
        reason = "EOL 또는 지원 종료 문서를 근거로 한 답변입니다."

    # 비교 질의에서 미회수 target 버전이 있으면 사유에 명시 (#108 — 회수율 coverage의 정직한 고지)
    missing = state.get("missing_versions", [])
    if missing and status == AnswerabilityStatus.PARTIALLY_ANSWERABLE:
        reason = f"{', '.join(missing)} 버전 문서를 찾지 못해 검색된 버전 기준으로만 답변합니다."

    # 보류 상태는 본문 비움. 단 CONFLICTING/OUTDATED는 "왜 보류됐는지" 근거 문서를 보여줘야 하므로 sources 유지(P1).
    if status in _HOLD_STATUSES:
        held_sources = [] if status == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE else sources
        return _result(
            answer_format=answer_format,
            answer=FiveElements(),
            answer_text="",
            sources=held_sources,
            status=status,
            reason=reason,
        )
    return _result(
        answer_format=answer_format,
        answer=five,
        answer_text=answer_text,
        sources=sources,
        status=status,
        reason=reason,
    )
