"""Retriever Agent 노드 — dense 검색 + 리랭킹 → SourceDocument.

async 노드이므로 그래프는 ainvoke로 실행해야 한다 (chat_service도 ainvoke 사용).

점수 분리 (#103, 설계 7.3): Cross-Encoder 원점수(raw, [0,1])는 τ 진단용으로 보존하고,
부스트(최신성·도메인·버전·권위)가 합산된 ranking 점수는 정렬·top-N 선별에만 쓴다.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import anyio

from app.agents.retriever.agentic import AgenticRetrievalFallbackError, run_agentic_search
from app.agents.retriever.rerank import apply_ranking_boosts, get_reranker
from app.agents.retriever.search import SearchFilters, search_with_mode
from app.agents.retriever.tools import SearchToolContext
from app.agents.state import AgentState, Domain, RetryAction, SourceDocument
from app.config import Settings, get_settings
from app.observability import langfuse_span
from app.rag.embedder import get_embedder
from app.rag.lineage import get_lineages

logger = logging.getLogger(__name__)

# (ranking_score, raw_rerank_score, payload) — raw는 리랭킹 미수행 시 0.0
RankedRow = tuple[float, float, dict]


@dataclass
class RetrievalDiagnostics:
    """Retriever A/B 평가용 진단값. 운영 AgentState에는 노출하지 않는다."""

    strategy: str
    fallback: str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_count: int = 0
    duplicate_calls: int = 0
    ranking_list_count: int = 0
    rrf_applied: bool = False
    rerank_fallback: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "fallback": self.fallback,
            "calls": self.calls,
            "tool_call_count": self.tool_call_count,
            "duplicate_calls": self.duplicate_calls,
            "ranking_list_count": self.ranking_list_count,
            "rrf_applied": self.rrf_applied,
            "rerank_fallback": self.rerank_fallback,
        }


async def retrieve_with_diagnostics(
    state: AgentState,
    *,
    settings: Settings | None = None,
) -> tuple[dict, RetrievalDiagnostics]:
    """운영 Retriever와 동일한 경로를 실행하고 평가용 진단값을 함께 반환한다."""
    settings = settings or get_settings()
    diagnostics = RetrievalDiagnostics(strategy=settings.retriever_strategy)
    refined = state["refined_query"]
    # 질의 도메인 집합(순서 우선). domains 키가 **아예 없을 때만** 구형 단수 domain으로 폴백.
    # (명시적 domains=[] — 예: Trust 재검색 초기화 — 은 그대로 빈 집합으로 존중, 단수로 복구 금지)
    domains_state = state.get("domains")
    if domains_state is None:
        single = state.get("domain")
        domains_state = [single] if single else []
    domains = _domain_values(domains_state)
    # Router가 질의에서 추출한 target 버전 (#108 — match 모드 부스트에 사용)
    target_versions = [str(v) for v in state.get("target_versions", [])]

    # 재검색 사다리 상태 소비 (#108 — trust가 채움. 1차 검색에서는 전부 기본값)
    retry_action = state.get("retry_action")
    filters = SearchFilters(
        version=state.get("version_filter") or None,
        pinned_doc_keys=tuple(state.get("pinned_doc_keys", [])),
        excluded_doc_keys=tuple(state.get("excluded_doc_keys", [])),
    )
    top_k = settings.retriever_top_k
    if retry_action == RetryAction.EXPAND_TOPICS:
        top_k *= 2  # 주제 확장: 후보 풀 확대 (설계 6장)

    effective_filters = None if filters.is_empty() else filters
    domain = domains[0] if domains else None
    hits = None
    if settings.retriever_strategy == "agentic":
        with langfuse_span(
            name="agentic_retriever",
            input={"query": refined, "domain": domain, "tenant_id": state.get("tenant_id", "")},
        ) as span:
            try:
                agentic_result = await run_agentic_search(
                    refined,
                    model=state.get("model", ""),
                    context=SearchToolContext(
                        tenant_id=state.get("tenant_id") or settings.auth_default_tenant,
                        domain=None if settings.retriever_domain_filter_mode == "soft" else domain,
                        filters=effective_filters,
                        top_k=top_k,
                        settings=settings,
                    ),
                )
                hits = agentic_result.hits
                diagnostics.calls = list(agentic_result.metadata.get("calls", []))
                diagnostics.tool_call_count = int(agentic_result.metadata.get("tool_call_count", 0))
                diagnostics.duplicate_calls = int(agentic_result.metadata.get("duplicate_calls", 0))
                diagnostics.ranking_list_count = int(agentic_result.metadata.get("ranking_list_count", 0))
                diagnostics.rrf_applied = bool(agentic_result.metadata.get("rrf_applied", False))
                if span is not None:
                    span.update(metadata={**agentic_result.metadata, "fallback": None})
            except AgenticRetrievalFallbackError as exc:
                logger.warning("Agentic Retriever 결과 없음 — deterministic 검색 폴백 (%s)", exc.reason)
                diagnostics.fallback = exc.reason
                if span is not None:
                    span.update(metadata={"fallback": exc.reason})
            except Exception as exc:
                logger.warning("Agentic Retriever 실패 — deterministic 검색 폴백", exc_info=True)
                diagnostics.fallback = type(exc).__name__
                if span is not None:
                    span.update(metadata={"fallback": type(exc).__name__})

    if hits is None:
        qvec = await get_embedder().embed_query(refined)
        # 필터용 domain은 대표(domains[0])만 — soft에선 무시되고 hard/hybrid에서만 쓰인다.
        hits = await search_with_mode(
            qvec,
            top_k,
            domain=domain,
            mode=settings.retriever_domain_filter_mode,
            query_text=refined,
            filters=effective_filters,
            settings=settings,
        )

    results = [(point.score, point.payload or {}) for point in hits]
    vec_score = {payload.get("chunk_id"): score for score, payload in results}
    candidates = [(payload.get("content", ""), payload) for _, payload in results]

    # 버전 부스트용 계보 배치 조회 (Qdrant facet, TTL 캐시) — 동기 함수라 스레드로 오프로드
    doc_keys = [payload.get("doc_key", "") or "" for _, payload in results]
    lineages = await anyio.to_thread.run_sync(partial(get_lineages, doc_keys, settings=settings))

    # rerank(외부 GPU remote 등)를 span으로 감싸 backend·0-hit·폴백·top score 관측 (비활성 no-op).
    fallback_reason: str | None = None
    with langfuse_span(
        name="rerank", input={"backend": settings.reranker_backend, "n_candidates": len(candidates)}
    ) as span:
        try:
            # CrossEncoder.predict는 CPU 동기 작업 → 스레드로 오프로드 (이벤트 루프 비차단)
            reranked = await anyio.to_thread.run_sync(get_reranker().rerank, refined, candidates)
            ranked: list[RankedRow] = [
                (apply_ranking_boosts(raw, payload, domains, lineages, target_versions, settings), raw, payload)
                for raw, payload in reranked
            ]
            ranked.sort(key=lambda item: item[0], reverse=True)  # ranking 점수로 재정렬 (raw는 진단용 보존)
        except ModuleNotFoundError:  # sentence-transformers 미설치 → 리랭커 비활성
            logger.warning(
                "리랭커 비활성 — sentence-transformers 미설치. vector score 순 폴백 (설치: make install-rerank)"
            )
            ranked = _vector_fallback(results, domains, lineages, target_versions, settings)
            fallback_reason = "module_missing"
        except Exception:  # 리랭커 로드/실행 실패(OOM·timeout·원격 5xx 등) → vector score 순 폴백
            logger.warning("리랭커 로드/실행 실패 — vector score 순으로 폴백", exc_info=True)
            ranked = _vector_fallback(results, domains, lineages, target_versions, settings)
            fallback_reason = "error"
        if span is not None:
            span.update(
                metadata={
                    "backend": settings.reranker_backend,
                    "n_hits": len(results),
                    "zero_hit": not results,
                    "n_candidates": len(candidates),
                    "fallback": fallback_reason,
                    "reranked": fallback_reason is None,
                    "top_raw_score": max((raw for _, raw, _ in ranked), default=0.0),
                }
            )

    docs = [
        _to_source_doc(payload, ranking_score, raw_score, vec_score.get(payload.get("chunk_id"), 0.0), settings)
        for ranking_score, raw_score, payload in ranked[: settings.retriever_top_n]
    ]
    diagnostics.rerank_fallback = fallback_reason is not None
    # 리랭커 폴백 여부를 Trust로 전달 — coverage 산정이 raw rerank τ 대신 검색점수 비율을 쓰게 한다 (#202)
    return (
        {"documents": docs, "rerank_fallback": fallback_reason is not None, "agent_trace": ["retriever"]},
        diagnostics,
    )


async def retrieve_node(state: AgentState) -> dict:
    """정제 쿼리로 검색·리랭킹해 top-N 출처 문서를 반환한다."""
    output, _ = await retrieve_with_diagnostics(state)
    return output


def _domain_values(domains: Sequence[Domain | str] | None) -> list[str]:
    """state의 domains(Domain enum / str 혼재)를 payload 비교용 str 리스트로 정규화한다."""
    if not domains:
        return []
    return [d.value if isinstance(d, Domain) else d for d in domains]


def _vector_fallback(
    results: list[tuple[float, dict]],
    domains: list[str],
    lineages: dict[str, frozenset[str]],
    target_versions: list[str],
    settings: Settings,
) -> list[RankedRow]:
    """리랭커 불가 시 vector score 순으로 정렬한다.

    Soft 정책 일관성 — 리랭커가 없어도 부스트 체인을 정렬 키(vector score)에 동일 적용한다.
    rerank/raw 점수 표기는 둘 다 0.0 유지(리랭킹 미수행 신호 — raw τ 진단도 동일하게 폴백 인지).
    """
    ordered = sorted(
        results,
        key=lambda item: apply_ranking_boosts(item[0], item[1], domains, lineages, target_versions, settings),
        reverse=True,
    )
    return [(0.0, 0.0, payload) for _, payload in ordered]


def _clean_url(url: str | None) -> str:
    """scheme(://) 뒤 중복 슬래시를 단일화 — #210 이전 적재분의 stale source_url('//wiki' 등) 교정(#225).

    payload의 source_url이 None/빈값일 수 있으므로 falsy는 빈 문자열로 안전 처리한다.
    """
    if not url or "://" not in url:
        return url or ""
    scheme, rest = url.split("://", 1)
    return f"{scheme}://{re.sub(r'/{2,}', '/', rest)}"


def _to_source_doc(
    payload: dict, ranking_score: float, raw_score: float, score: float, settings: Settings
) -> SourceDocument:
    return SourceDocument(
        title=payload.get("page_title", ""),
        url=_clean_url(payload.get("source_url", "")),
        space_key=payload.get("space_key", ""),
        content_snippet=payload.get("content", "")[: settings.snippet_max_chars],
        score=score,
        rerank_score=ranking_score,
        raw_rerank_score=raw_score,
        page_id=payload.get("page_id", ""),
        last_modified=payload.get("last_modified", ""),
        hash=payload.get("hash", ""),
        chunk_id=payload.get("chunk_id", "") or "",
        parent_id=payload.get("parent_id", "") or "",
        site=payload.get("site", "") or "",
        product_version=payload.get("product_version", "") or "",
        doc_key=payload.get("doc_key", "") or "",
        is_eol=bool(payload.get("is_eol", False)),
    )
