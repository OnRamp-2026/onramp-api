"""Retriever Agent 노드 — dense 검색 + 리랭킹 → SourceDocument.

async 노드이므로 그래프는 ainvoke로 실행해야 한다 (chat_service도 ainvoke 사용).

점수 분리 (#103, 설계 7.3): Cross-Encoder 원점수(raw, [0,1])는 τ 진단용으로 보존하고,
부스트(최신성·도메인·버전·권위)가 합산된 ranking 점수는 정렬·top-N 선별에만 쓴다.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from functools import partial

import anyio

from app.agents.retriever.rerank import apply_ranking_boosts, get_reranker, limit_ranking_boost_spread
from app.agents.retriever.search import SearchFilters, search_with_mode
from app.agents.state import AgentState, Domain, RetryAction, SourceDocument
from app.config import Settings, get_settings
from app.observability import langfuse_span
from app.rag.embedder import get_embedder
from app.rag.lineage import get_lineages

logger = logging.getLogger(__name__)

# (ranking_score, raw_rerank_score, payload) — raw는 리랭킹 미수행 시 0.0
RankedRow = tuple[float, float, dict]


async def retrieve_node(state: AgentState) -> dict:
    """설정된 strategy로 검색·리랭킹해 top-N 출처 문서를 반환한다."""
    settings = get_settings()
    if state.get("retriever_strategy") == "single_agentic" or settings.retriever_strategy == "single_agentic":
        return await _retrieve_single_agentic(state, settings)
    return await _retrieve_deterministic(state, settings)


async def _retrieve_deterministic(state: AgentState, settings: Settings) -> dict:
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

    qvec = await get_embedder().embed_query(refined)
    # 필터용 domain은 대표(domains[0])만 — soft에선 무시되고 hard/hybrid에서만 쓰인다.
    hits = await search_with_mode(
        qvec,
        top_k,
        domain=(domains[0] if domains else None),
        mode=settings.retriever_domain_filter_mode,
        query_text=refined,
        tenant_id=state.get("tenant_id"),
        filters=None if filters.is_empty() else filters,
        settings=settings,
    )

    results = [(point.score, point.payload or {}) for point in hits]
    docs, fallback_reason = await _rank_results(refined, results, domains, target_versions, settings)
    return {"documents": docs, "rerank_fallback": fallback_reason is not None, "agent_trace": ["retriever"]}


async def _retrieve_single_agentic(state: AgentState, settings: Settings) -> dict:
    from app.agents.retriever.agentic import run_agentic_step
    from app.agents.state import RetrievalPhase

    step = await run_agentic_step(state, settings)
    if step.get("retrieval_phase") == RetrievalPhase.COMPLETE:
        return step
    candidates = step.get("retrieval_candidates", [])
    results = [(candidate.search_score, candidate.payload) for candidate in candidates]
    domains = _domain_values(state.get("domains", []))
    target_versions = [str(v) for v in state.get("target_versions", [])]
    query = step.get("previous_queries", [])[-1] if step.get("previous_queries") else state.get("query", "")
    docs, fallback_reason = await _rank_results(query, results, domains, target_versions, settings)
    return {
        **step,
        "documents": docs,
        "rerank_fallback": fallback_reason is not None,
    }


async def _rank_results(
    query: str,
    results: list[tuple[float, dict]],
    domains: list[str],
    target_versions: list[str],
    settings: Settings,
) -> tuple[list[SourceDocument], str | None]:
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
            reranked = await anyio.to_thread.run_sync(get_reranker().rerank, query, candidates)
            ranked: list[RankedRow] = [
                (apply_ranking_boosts(raw, payload, domains, lineages, target_versions, settings), raw, payload)
                for raw, payload in reranked
            ]
            ranked = limit_ranking_boost_spread(ranked, max_spread=settings.rank_boost_max_spread)
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

    selected = ranked[: settings.retriever_top_n]
    selected_ids = {str(payload.get("chunk_id") or "") for _, _, payload in selected}
    selected.extend(
        row
        for row in ranked[settings.retriever_top_n :]
        if str(row[2].get("chunk_id") or "").startswith("document:")
        and str(row[2].get("chunk_id") or "") not in selected_ids
    )
    docs = [
        _to_source_doc(
            payload,
            ranking_score,
            raw_score,
            vec_score.get(payload.get("chunk_id"), 0.0),
            query,
            settings,
        )
        for ranking_score, raw_score, payload in selected
    ]
    return docs, fallback_reason


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


def _select_snippet(content: str, query: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    terms = {term.casefold() for term in re.findall(r"[0-9A-Za-z가-힣_-]{2,}", query)}
    lowered = content.casefold()
    positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
    if not positions:
        return content[:max_chars]
    candidates: list[tuple[int, int, str]] = []
    for position in positions:
        start = max(0, min(position - max_chars // 3, len(content) - max_chars))
        snippet = content[start : start + max_chars]
        score = sum(term in snippet.casefold() for term in terms)
        candidates.append((score, -start, snippet))
    return max(candidates)[2]


def _to_source_doc(
    payload: dict,
    ranking_score: float,
    raw_score: float,
    score: float,
    query: str,
    settings: Settings,
) -> SourceDocument:
    return SourceDocument(
        title=payload.get("page_title", ""),
        url=_clean_url(payload.get("source_url", "")),
        space_key=payload.get("space_key", ""),
        source=payload.get("source", "") or "",
        content_snippet=_select_snippet(payload.get("content", ""), query, settings.snippet_max_chars),
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
