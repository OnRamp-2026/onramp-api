"""Retriever Agent 노드 — dense 검색 + 리랭킹 → SourceDocument.

async 노드이므로 그래프는 ainvoke로 실행해야 한다 (chat_service도 ainvoke 사용).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import anyio

from app.agents.retriever.rerank import apply_domain_weight, apply_metadata_weight, get_reranker
from app.agents.retriever.search import search_with_mode
from app.agents.state import AgentState, Domain, SourceDocument
from app.config import Settings, get_settings
from app.rag.embedder import get_embedder

logger = logging.getLogger(__name__)


async def retrieve_node(state: AgentState) -> dict:
    """정제 쿼리로 검색·리랭킹해 top-N 출처 문서를 반환한다."""
    settings = get_settings()
    refined = state["refined_query"]
    # 질의 도메인 집합(순서 우선). domains 키가 **아예 없을 때만** 구형 단수 domain으로 폴백.
    # (명시적 domains=[] — 예: Trust 재검색 초기화 — 은 그대로 빈 집합으로 존중, 단수로 복구 금지)
    domains_state = state.get("domains")
    if domains_state is None:
        single = state.get("domain")
        domains_state = [single] if single else []
    domains = _domain_values(domains_state)

    qvec = await get_embedder().embed_query(refined)
    # 필터용 domain은 대표(domains[0])만 — soft에선 무시되고 hard/hybrid에서만 쓰인다.
    hits = await search_with_mode(
        qvec,
        settings.retriever_top_k,
        domain=(domains[0] if domains else None),
        mode=settings.retriever_domain_filter_mode,
        settings=settings,
    )

    results = [(point.score, point.payload or {}) for point in hits]
    vec_score = {payload.get("chunk_id"): score for score, payload in results}
    candidates = [(payload.get("content", ""), payload) for _, payload in results]

    try:
        # CrossEncoder.predict는 CPU 동기 작업 → 스레드로 오프로드 (이벤트 루프 비차단)
        reranked = await anyio.to_thread.run_sync(get_reranker().rerank, refined, candidates)
        ranked = [
            (apply_domain_weight(apply_metadata_weight(score, payload, settings), payload, domains, settings), payload)
            for score, payload in reranked
        ]
        ranked.sort(key=lambda item: item[0], reverse=True)  # 가중 반영 후 재정렬
    except ModuleNotFoundError:  # sentence-transformers 미설치 → 리랭커 비활성
        logger.warning("리랭커 비활성 — sentence-transformers 미설치. vector score 순 폴백 (설치: make install-rerank)")
        ranked = _vector_fallback(results, domains, settings)
    except Exception:  # 리랭커 로드/실행 실패(OOM 등) → vector score 순 폴백
        logger.warning("리랭커 로드/실행 실패 — vector score 순으로 폴백", exc_info=True)
        ranked = _vector_fallback(results, domains, settings)

    docs = [
        _to_source_doc(payload, rerank_score, vec_score.get(payload.get("chunk_id"), 0.0), settings)
        for rerank_score, payload in ranked[: settings.retriever_top_n]
    ]
    return {"documents": docs, "agent_trace": ["retriever"]}


def _domain_values(domains: Sequence[Domain | str] | None) -> list[str]:
    """state의 domains(Domain enum / str 혼재)를 payload 비교용 str 리스트로 정규화한다."""
    if not domains:
        return []
    return [d.value if isinstance(d, Domain) else d for d in domains]


def _vector_fallback(
    results: list[tuple[float, dict]], domains: list[str], settings: Settings
) -> list[tuple[float, dict]]:
    """리랭커 불가 시 vector score 순으로 정렬한다.

    Soft 정책 일관성 — 리랭커가 없어도 도메인 가산을 정렬 키(vector score)에 반영한다.
    rerank_score 표기는 0.0 유지(리랭킹 미수행 신호).
    """
    ordered = sorted(results, key=lambda item: apply_domain_weight(item[0], item[1], domains, settings), reverse=True)
    return [(0.0, payload) for _, payload in ordered]


def _to_source_doc(payload: dict, rerank_score: float, score: float, settings: Settings) -> SourceDocument:
    return SourceDocument(
        title=payload.get("page_title", ""),
        url=payload.get("source_url", ""),
        space_key=payload.get("space_key", ""),
        content_snippet=payload.get("content", "")[: settings.snippet_max_chars],
        score=score,
        rerank_score=rerank_score,
        page_id=payload.get("page_id", ""),
        last_modified=payload.get("last_modified", ""),
        hash=payload.get("hash", ""),
    )
