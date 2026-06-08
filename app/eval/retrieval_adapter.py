"""검색 평가 어댑터 — query를 우리 검색 경로로 흘려 ranked chunk_id를 반환한다.

`retrieve_node`(app/agents/retriever/node.py)의 로직을 그대로 미러하되,
SourceDocument 대신 평가용 chunk_id 리스트/점수를 돌려준다.
LLM-free(임베딩 검색만) → Router를 거치지 않고 raw query를 그대로 검색에 투입한다.

mode:
    "dense"  — dense_search 결과를 vector score 순으로 top_n
    "rerank" — Cross-Encoder 재정렬 + 최신성 가중 후 top_n (운영 경로와 동일)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import anyio

from app.agents.retriever.rerank import apply_metadata_weight, get_reranker
from app.agents.retriever.search import dense_search
from app.config import Settings, get_settings
from app.rag.embedder import get_embedder

logger = logging.getLogger(__name__)

Mode = Literal["dense", "rerank"]


@dataclass(frozen=True)
class RetrievalResult:
    """평가용 검색 결과."""

    chunk_ids: list[str]  # top_n, 순위 순
    top_score: float  # 1위 점수 (rerank 모드=가중 rerank 점수 / dense 모드=vector score)
    n: int  # 반환된 문서 수


async def retrieve_for_eval(
    query: str,
    *,
    mode: Mode,
    domain: str | None = None,
    top_k: int | None = None,
    top_n: int | None = None,
    settings: Settings | None = None,
) -> RetrievalResult:
    """query를 검색해 ranked chunk_id와 1위 점수를 반환한다 (retrieve_node 미러)."""
    settings = settings or get_settings()
    top_k = settings.retriever_top_k if top_k is None else top_k
    top_n = settings.retriever_top_n if top_n is None else top_n
    if top_k <= 0 or top_n <= 0:
        raise ValueError(f"top_k/top_n 은 1 이상이어야 합니다: top_k={top_k}, top_n={top_n}")

    qvec = await get_embedder().embed_query(query)
    hits = await dense_search(qvec, top_k, domain=domain, settings=settings)
    if not hits and domain:  # 도메인 과필터 0건 → 무필터 재검색 (recall 안전, retrieve_node 동일)
        hits = await dense_search(qvec, top_k, domain=None, settings=settings)

    results = [(point.score, point.payload or {}) for point in hits]

    if mode == "dense":
        ranked = sorted(results, key=lambda item: item[0], reverse=True)
    else:  # rerank
        candidates = [(payload.get("content", ""), payload) for _, payload in results]
        try:
            reranked = await anyio.to_thread.run_sync(get_reranker().rerank, query, candidates)
            ranked = [(apply_metadata_weight(score, payload, settings), payload) for score, payload in reranked]
            ranked.sort(key=lambda item: item[0], reverse=True)
        except Exception as exc:  # 리랭커 실패 → vector score 순 폴백 (retrieve_node 동일)
            logger.warning("리랭커 실패로 dense 폴백: %s", exc, exc_info=True)
            ranked = sorted(results, key=lambda item: item[0], reverse=True)

    top = ranked[:top_n]
    chunk_ids = [payload.get("chunk_id", "") for _, payload in top if payload.get("chunk_id")]
    top_score = top[0][0] if top else 0.0
    return RetrievalResult(chunk_ids=chunk_ids, top_score=top_score, n=len(chunk_ids))


async def ranked_chunk_ids(
    query: str,
    *,
    mode: Mode,
    domain: str | None = None,
    top_k: int | None = None,
    top_n: int | None = None,
    settings: Settings | None = None,
) -> list[str]:
    """검색 지표용 — ranked chunk_id 리스트만 반환."""
    result = await retrieve_for_eval(query, mode=mode, domain=domain, top_k=top_k, top_n=top_n, settings=settings)
    return result.chunk_ids


def predicted_answerable(result: RetrievalResult, *, floor: float, min_docs: int) -> bool:
    """결정론 answerable 예측 — 1위 점수가 floor 이상이고 문서 수가 min_docs 이상.

    floor(τ)는 reranker 점수 분포에 의존 → 베이스라인 측정 후 보정한다.
    이 신호를 #B Trust 재검색 트리거가 재사용한다.
    """
    return result.n >= min_docs and result.top_score >= floor
