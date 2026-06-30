"""Single Agentic Retriever policy loop."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from time import perf_counter
from typing import Any

from app.agents.answer.answerability import ANSWERABLE_THRESHOLD
from app.agents.retriever.tools import TOOL_SCHEMAS, SearchToolContext, execute_tool
from app.agents.state import AgentState, RetrievalCandidate, RetrievalPhase, ToolTrace
from app.config import Settings
from app.middleware.error_handler import LLMError
from app.observability import agent_metrics, score_current_trace
from app.services.llm_selector import ToolCall, ToolResponse, call_llm_with_tools

logger = logging.getLogger(__name__)


async def _call_with_retry(messages: list[dict[str, Any]], *, model: str, settings: Settings) -> ToolResponse:
    """tool-call 판단 LLM을 bounded retry로 호출 — 일시 실패(timeout/5xx/rate-limit)를 회복한다.

    검색 인프라(execute_tool)는 이미 graceful degrade(빈 결과)라 재시도 대상이 아니다.
    재시도를 모두 소진하면 마지막 예외를 올려 호출부가 기존 hybrid_search fallback으로
    퇴화하게 한다 — 즉 복원력을 더하되 기존 안전망은 그대로 유지한다.
    """
    attempts = settings.single_agentic_llm_max_retries + 1
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return await call_llm_with_tools(
                messages,
                TOOL_SCHEMAS,
                model=model,
                timeout=settings.single_agentic_llm_timeout_s,
                settings=settings,
            )
        except LLMError as exc:
            last_exc = exc
            if attempt + 1 >= attempts:
                break
            backoff = settings.single_agentic_llm_backoff_base_s * (2**attempt)
            logger.warning(
                "Single Agentic tool-call LLM 일시 실패 — 재시도 %d/%d (backoff %.2fs)",
                attempt + 1,
                attempts - 1,
                backoff,
            )
            if backoff > 0:
                await asyncio.sleep(backoff)
    raise last_exc if last_exc is not None else LLMError("tool-call LLM 호출에 실패했습니다")


def _emit_tool_observation(traces: list[ToolTrace], merged: list[RetrievalCandidate]) -> None:
    """ToolTrace 요약을 현재 Langfuse trace에 score로 부착한다 (비활성이면 no-op).

    에이전트의 자율 tool 선택·source·결과 수·fallback을 trace에서 바로 보이게 해
    "왜 이 도구를 골랐고 언제 fallback했는지"를 운영 중에 디버깅할 수 있게 한다.
    """
    if not traces:
        return
    fallbacks = sum(1 for trace in traces if trace.fallback)
    summary = "; ".join(
        f"{trace.tool}{'/' + trace.source if trace.source else ''}→{trace.n_results}"
        + (f"[{trace.fallback}]" if trace.fallback else "")
        for trace in traces
    )
    score_current_trace(name="agentic_tool_calls", value=float(len(traces)), comment=summary)
    score_current_trace(name="agentic_tool_fallbacks", value=float(fallbacks))
    score_current_trace(name="agentic_candidates", value=float(len(merged)))
    # 운영 모니터링용 집계 카운터 (#284) — Langfuse(요청별 디버깅)와 보완.
    agent_metrics.record_agentic_step(
        [trace.tool for trace in traces],
        fallbacks=fallbacks,
        retried=any(trace.retry > 0 for trace in traces),
    )


SYSTEM_PROMPT = """너는 OnRamp Single Agentic Retriever다.
원문 질문에 답할 근거를 찾고, Trust 평가를 참고해 검색을 종료하거나 재검색한다.
- 첫 진입에서는 반드시 hybrid_search 또는 hybrid_search_by_source를 호출한다.
- 기본은 hybrid_search(전체 출처)다. source를 추측하지 말 것 — 질의가 콘텐츠 타입을 분명히 드러낼 때만 hybrid_search_by_source로 좁힌다:
  · github: 이슈/PR/커밋/코드/README/설정파일(values.yaml·Jenkinsfile·CI)/환경변수/인프라·배포(gitops·k8s)/에러·증상·디버깅(OOM·CrashLoop·500)/구체적 코드·설정 수정
  · confluence: 회의·회의록/보고서/온보딩/매뉴얼·가이드/정책·규칙/용어·정의/기획서/프로세스·절차/FAQ/아키텍처 설명
- 콘텐츠 타입이 불명확하거나 source 제한 결과가 부족하면 hybrid_search(전체)를 쓴다. 도메인만으로 source를 단정하지 않는다.
- opensearch_get_document는 incident 질의이며 앞선 검색 결과의 doc_id를 확인한 뒤에만 사용한다.
- previous_queries와 같은 검색어를 반복하지 않는다.
- 현재 evidence로 질문에 답할 수 있으면 tool을 호출하지 않는다.
- 한 단계에서 tool은 최대 두 개만 호출한다."""


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _tool_identity(tool: str, query: str, source: str = "") -> tuple[str, str, str]:
    return tool, _normalize(query).casefold(), source.casefold()


def _attempted_tool_identities(state: AgentState, existing: list[RetrievalCandidate]) -> set[tuple[str, str, str]]:
    attempted = {_tool_identity(trace.tool, trace.query, trace.source) for trace in state.get("tool_trace", [])}
    for candidate in existing:
        source = str(candidate.payload.get("source") or "") if candidate.tool_name == "hybrid_search_by_source" else ""
        attempted.add(_tool_identity(candidate.tool_name, candidate.query, source))
    return attempted


def _expand_initial_source_search(calls: list[ToolCall], *, limit: int) -> list[ToolCall]:
    expanded: list[ToolCall] = []
    for call in calls:
        if len(expanded) >= limit:
            break
        expanded.append(call)
        if call.name != "hybrid_search_by_source" or len(expanded) >= limit:
            continue
        query = _normalize(str(call.arguments.get("query") or ""))
        if query:
            expanded.append(
                ToolCall(
                    id=f"{call.id}:global",
                    name="hybrid_search",
                    arguments={"query": query},
                )
            )
    return expanded


def _domains(state: AgentState) -> tuple[str, ...]:
    return tuple(getattr(domain, "value", domain) for domain in state.get("domains", []))


def _can_complete_without_llm(state: AgentState, existing: list[RetrievalCandidate]) -> bool:
    trust = state.get("trust_score")
    gate = state.get("gate_flags")
    blocked = bool(gate and (gate.conflicting or gate.deprecated_only or gate.sensitive_block))
    return bool(existing and trust and trust.overall >= ANSWERABLE_THRESHOLD and not blocked)


def _evidence_prompt(state: AgentState) -> str:
    documents = [
        {
            "page_id": doc.page_id,
            "title": doc.title,
            "source": doc.source,
            "product_version": doc.product_version or None,
            "version_fit": doc.version_fit,
            "score": doc.rerank_score,
            "snippet": doc.content_snippet,
        }
        for doc in state.get("documents", [])
    ]
    trust = state.get("trust_score")
    gate = state.get("gate_flags")
    return json.dumps(
        {
            "query": state.get("query", ""),
            "domains": _domains(state),
            "target_versions": state.get("target_versions", []),
            "previous_queries": state.get("previous_queries", []),
            "retry_count": state.get("retry_count", 0),
            "trust_score": vars(trust) if trust else None,
            "gate_flags": vars(gate) if gate else None,
            "missing_versions": state.get("missing_versions", []),
            "documents": documents,
        },
        ensure_ascii=False,
    )


def merge_candidates(
    existing: list[RetrievalCandidate],
    additions: list[RetrievalCandidate],
    *,
    limit: int,
) -> list[RetrievalCandidate]:
    merged = {candidate.chunk_id: candidate for candidate in existing}
    for candidate in additions:
        current = merged.get(candidate.chunk_id)
        if current is None or candidate.search_score > current.search_score:
            merged[candidate.chunk_id] = candidate
    return sorted(merged.values(), key=lambda item: item.search_score, reverse=True)[:limit]


async def run_agentic_step(state: AgentState, settings: Settings) -> dict[str, Any]:
    tenant_id = state.get("tenant_id", "")
    if not tenant_id:
        raise ValueError("single_agentic 검색에는 tenant_id가 필요합니다")
    existing = list(state.get("retrieval_candidates", []))
    if _can_complete_without_llm(state, existing):
        return {"retrieval_phase": RetrievalPhase.COMPLETE, "agent_trace": ["retriever"]}
    candidate_doc_ids = frozenset(
        str(candidate.payload.get("page_id") or "") for candidate in existing if candidate.payload.get("page_id")
    )
    context = SearchToolContext(
        tenant_id=tenant_id,
        domains=_domains(state),
        candidate_doc_ids=candidate_doc_ids,
        filters=None,
        top_k=settings.retriever_top_k,
        settings=settings,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _evidence_prompt(state)},
    ]
    fallback = ""
    try:
        response = await _call_with_retry(messages, model=state.get("model", ""), settings=settings)
        calls = response.tool_calls[: settings.single_agentic_max_tools_per_step]
    except Exception as exc:
        logger.warning("Single Agentic Retriever LLM 실패 — hybrid fallback", exc_info=True)
        calls = [ToolCall(id="fallback", name="hybrid_search", arguments={"query": state.get("query", "")})]
        fallback = type(exc).__name__

    first_step = not existing
    if first_step:
        calls = _expand_initial_source_search(calls, limit=settings.single_agentic_max_tools_per_step)
    if not calls and first_step:
        calls = [ToolCall(id="fallback", name="hybrid_search", arguments={"query": state.get("query", "")})]
        fallback = "missing_initial_tool"
    if not calls:
        return {"retrieval_phase": RetrievalPhase.COMPLETE, "agent_trace": ["retriever"]}

    attempted = _attempted_tool_identities(state, existing)
    additions: list[RetrievalCandidate] = []
    traces: list[ToolTrace] = []
    queries: list[str] = []
    for call in calls:
        query = _normalize(str(call.arguments.get("query") or ""))
        source = str(call.arguments.get("source") or "")
        identity_query = query or str(call.arguments.get("doc_id") or "")
        identity = _tool_identity(call.name, identity_query, source)
        if not identity_query or identity in attempted:
            continue
        started = perf_counter()
        try:
            hits = await execute_tool(call.name, call.arguments, context=context)
            tool_fallback = fallback
        except Exception as exc:
            logger.warning("Single Agentic tool 실패: %s", call.name, exc_info=True)
            hits = []
            tool_fallback = type(exc).__name__
        traces.append(
            ToolTrace(
                tool=call.name,
                query=identity_query,
                source=source,
                n_results=len(hits),
                retry=state.get("retry_count", 0),
                latency_ms=(perf_counter() - started) * 1000,
                fallback=tool_fallback,
            )
        )
        attempted.add(identity)
        if query:
            queries.append(query)
        for hit in hits:
            payload = hit.payload or {}
            additions.append(
                RetrievalCandidate(
                    chunk_id=str(payload.get("chunk_id") or hit.id),
                    payload=payload,
                    search_score=float(hit.score),
                    tool_name=call.name,
                    query=query,
                )
            )
    if existing and not traces:
        return {"retrieval_phase": RetrievalPhase.COMPLETE, "agent_trace": ["retriever"]}
    merged = merge_candidates(existing, additions, limit=settings.single_agentic_max_candidates)
    retry_count = state.get("retry_count", 0) + (0 if first_step else 1)
    _emit_tool_observation(traces, merged)
    return {
        "retrieval_candidates": merged,
        "previous_queries": queries,
        "tool_trace": traces,
        "retry_count": retry_count,
        "retrieval_phase": RetrievalPhase.SEARCHED,
        "agent_trace": ["retriever"],
    }
