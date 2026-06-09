"""Retriever Agent 노드 — dense 검색 + 리랭킹 → SourceDocument.

async 노드이므로 그래프는 ainvoke로 실행해야 한다 (chat_service도 ainvoke 사용).
"""

from __future__ import annotations

import logging

import anyio
from qdrant_client.models import ScoredPoint

from app.agents.retriever.rerank import apply_domain_weight, apply_metadata_weight, get_reranker
from app.agents.retriever.search import dense_search
from app.agents.state import AgentState, Domain, SourceDocument
from app.config import Settings, get_settings
from app.rag.embedder import get_embedder

logger = logging.getLogger(__name__)


async def retrieve_node(state: AgentState) -> dict:
    """정제 쿼리로 검색·리랭킹해 top-N 출처 문서를 반환한다."""
    settings = get_settings()
    refined = state["refined_query"]
    domain = _domain_value(state.get("domain"))

    qvec = await get_embedder().embed_query(refined)
    hits = await dense_search(qvec, settings.retriever_top_k, domain=domain)
    # filtered 결과가 저품질이면(0건 또는 최고 score 미달) 무필터 검색을 합쳐 recall을 보완한다.
    if domain and settings.retriever_domain_expand_low_quality and _is_low_quality(hits, settings):
        extra = await dense_search(qvec, settings.retriever_top_k, domain=None)
        hits = _merge_hits(hits, extra)

    results = [(point.score, point.payload or {}) for point in hits]
    vec_score = {payload.get("chunk_id"): score for score, payload in results}
    candidates = [(payload.get("content", ""), payload) for _, payload in results]

    try:
        # CrossEncoder.predict는 CPU 동기 작업 → 스레드로 오프로드 (이벤트 루프 비차단)
        reranked = await anyio.to_thread.run_sync(get_reranker().rerank, refined, candidates)
        ranked = [
            (apply_domain_weight(apply_metadata_weight(score, payload, settings), payload, domain, settings), payload)
            for score, payload in reranked
        ]
        ranked.sort(key=lambda item: item[0], reverse=True)  # 가중 반영 후 재정렬
    except ModuleNotFoundError:  # sentence-transformers 미설치 → 리랭커 비활성
        logger.warning("리랭커 비활성 — sentence-transformers 미설치. vector score 순 폴백 (설치: make install-rerank)")
        ranked = _vector_fallback(results)
    except Exception:  # 리랭커 로드/실행 실패(OOM 등) → vector score 순 폴백
        logger.warning("리랭커 로드/실행 실패 — vector score 순으로 폴백", exc_info=True)
        ranked = _vector_fallback(results)

    docs = [
        _to_source_doc(payload, rerank_score, vec_score.get(payload.get("chunk_id"), 0.0), settings)
        for rerank_score, payload in ranked[: settings.retriever_top_n]
    ]
    return {"documents": docs, "agent_trace": ["retriever"]}


def _domain_value(domain: Domain | str | None) -> str | None:
    """state의 domain(Domain enum / str / None)을 payload 필터용 str로 정규화한다."""
    if domain is None:
        return None
    return domain.value if isinstance(domain, Domain) else domain


def _is_low_quality(hits: list[ScoredPoint], settings: Settings) -> bool:
    """결과가 없거나 최고 score가 retriever_domain_min_score 미만이면 저품질로 본다."""
    if not hits:
        return True
    return max(point.score for point in hits) < settings.retriever_domain_min_score


def _merge_hits(primary: list[ScoredPoint], extra: list[ScoredPoint]) -> list[ScoredPoint]:
    """두 검색 결과를 point.id로 합치고 중복은 높은 score를 남긴다."""
    by_id: dict = {point.id: point for point in primary}
    for point in extra:
        existing = by_id.get(point.id)
        if existing is None or point.score > existing.score:
            by_id[point.id] = point
    return sorted(by_id.values(), key=lambda point: point.score, reverse=True)


def _vector_fallback(results: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
    """리랭커 불가 시 vector score 순으로 정렬 (rerank_score는 0.0으로 표기)."""
    ordered = sorted(results, key=lambda item: item[0], reverse=True)
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
