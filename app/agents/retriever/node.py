"""Retriever Agent 노드 — dense 검색 + 리랭킹 → SourceDocument.

async 노드이므로 그래프는 ainvoke로 실행해야 한다 (chat_service도 ainvoke 사용).
"""

from __future__ import annotations

import logging

import anyio

from app.agents.retriever.rerank import apply_metadata_weight, get_reranker
from app.agents.retriever.search import dense_search
from app.agents.state import AgentState, SourceDocument
from app.config import Settings, get_settings
from app.rag.embedder import get_embedder

logger = logging.getLogger(__name__)


async def retrieve_node(state: AgentState) -> dict:
    """정제 쿼리로 검색·리랭킹해 top-N 출처 문서를 반환한다."""
    settings = get_settings()
    refined = state["refined_query"]
    domain = state.get("domain")

    qvec = await get_embedder().embed_query(refined)
    hits = await dense_search(qvec, settings.retriever_top_k, domain=domain)
    if not hits and domain:  # 도메인 과필터로 0건 → 무필터 재검색 (recall 안전)
        hits = await dense_search(qvec, settings.retriever_top_k, domain=None)

    results = [(point.score, point.payload or {}) for point in hits]
    vec_score = {payload.get("chunk_id"): score for score, payload in results}
    candidates = [(payload.get("content", ""), payload) for _, payload in results]

    try:
        # CrossEncoder.predict는 CPU 동기 작업 → 스레드로 오프로드 (이벤트 루프 비차단)
        reranked = await anyio.to_thread.run_sync(get_reranker().rerank, refined, candidates)
        ranked = [(apply_metadata_weight(score, payload, settings), payload) for score, payload in reranked]
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
    )
