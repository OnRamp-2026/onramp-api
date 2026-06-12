"""검색 평가 어댑터 — query를 우리 검색 경로로 흘려 ranked chunk_id를 반환한다.

`retrieve_node`(app/agents/retriever/node.py)와 같은 검색 코어(search_with_mode)를 써서
SourceDocument 대신 평가용 chunk_id 리스트/점수를 돌려준다.
LLM-free(임베딩 검색만) → Router를 거치지 않고 raw query를 그대로 검색에 투입한다.

mode (랭킹 방식):
    "dense"  — vector score + 도메인 가산 순으로 top_n (운영 _vector_fallback과 동일한 soft 정책)
    "rerank" — Cross-Encoder 재정렬 + 최신성 가중 + 도메인 일치 가산 (운영 경로와 동일)
filter_mode (도메인 필터 전략, None이면 config 기본=운영과 동일):
    "hard" / "hybrid" / "soft" — search_with_mode 참고
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import anyio

from app.agents.retriever.rerank import apply_domain_weight, apply_metadata_weight, get_reranker
from app.agents.retriever.search import FilterMode, search_with_mode
from app.config import Settings, get_settings
from app.rag.embedder import get_embedder

logger = logging.getLogger(__name__)

Mode = Literal["dense", "rerank"]


def _soft_ranked(
    results: list[tuple[float, dict]], query_domains: list[str] | None, settings: Settings
) -> list[tuple[float, dict]]:
    """vector score에 도메인 가산을 적용해 정렬 — 운영 retrieve_node._vector_fallback과 동일한 soft 정책.

    query_domains가 비거나 문서 domain과 불일치면 가산 0 → 순수 vector score 정렬과 동치.
    (이게 빠지면 soft A/B가 '무필터 dense'만 측정해 도메인 가산 효과를 누락한다.)
    """
    scored = [(apply_domain_weight(score, payload, query_domains, settings), payload) for score, payload in results]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


async def _rerank_base(query: str, results: list[tuple[float, dict]], settings: Settings) -> list[tuple[float, dict]]:
    """rerank + 최신성 가중까지(**도메인 가산 전**) base (score, payload). 리랭커 실패 시 vector score 그대로.

    도메인 가산 전이라 도메인 무관 — 같은 base에 도메인만 달리 가산하면 paired A/B를 만들 수 있다.
    """
    candidates = [(payload.get("content", ""), payload) for _, payload in results]
    try:
        reranked = await anyio.to_thread.run_sync(get_reranker().rerank, query, candidates)
        return [(apply_metadata_weight(score, payload, settings), payload) for score, payload in reranked]
    except Exception as exc:  # 리랭커 실패 → vector score 폴백 (retrieve_node._vector_fallback 동일)
        logger.warning("리랭커 실패로 dense 폴백: %s", exc, exc_info=True)
        return list(results)


async def base_soft_candidates(query: str, *, top_k: int, settings: Settings | None = None) -> list[tuple[float, dict]]:
    """soft A/B용 — 도메인 가산 **전** base (score, payload)를 1회 확보(검색+리랭크+최신성 가중).

    soft라 도메인 필터 없음(domain=None → 후보 도메인 독립). 같은 base에 도메인만 달리 가산해
    paired A/B를 만든다(검색·리랭크를 A/B 각각 2번 하지 않음 → secondary 외 변동 제거).
    """
    settings = settings or get_settings()
    if top_k <= 0:
        raise ValueError(f"top_k 는 1 이상이어야 합니다: {top_k}")
    qvec = await get_embedder().embed_query(query)
    hits = await search_with_mode(qvec, top_k, domain=None, mode="soft", settings=settings)
    results = [(point.score, point.payload or {}) for point in hits]
    return await _rerank_base(query, results, settings)


def rank_chunk_ids_from_base(
    base: list[tuple[float, dict]], domains: list[str] | None, settings: Settings, top_n: int
) -> list[str]:
    """base 후보에 **도메인 가산만** 적용·정렬·top_n → chunk_id (paired A/B arm, Qdrant 불필요·결정론)."""
    if top_n <= 0:
        raise ValueError(f"top_n 는 1 이상이어야 합니다: {top_n}")
    ranked = _soft_ranked(base, domains, settings)[:top_n]
    return [payload.get("chunk_id", "") for _, payload in ranked if payload.get("chunk_id")]


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
    domains: list[str] | None = None,
    filter_mode: FilterMode | None = None,
    top_k: int | None = None,
    top_n: int | None = None,
    settings: Settings | None = None,
) -> RetrievalResult:
    """query를 검색해 ranked chunk_id와 1위 점수를 반환한다 (retrieve_node와 동일 코어).

    domains: 질의 도메인 집합(순서 우선). soft 가산이 문서 단일 domain을 이 집합과 비교한다.
    """
    settings = settings or get_settings()
    top_k = settings.retriever_top_k if top_k is None else top_k
    top_n = settings.retriever_top_n if top_n is None else top_n
    if top_k <= 0 or top_n <= 0:
        raise ValueError(f"top_k/top_n 은 1 이상이어야 합니다: top_k={top_k}, top_n={top_n}")
    effective_filter = filter_mode if filter_mode is not None else settings.retriever_domain_filter_mode

    qvec = await get_embedder().embed_query(query)
    # 필터용 domain은 대표(domains[0])만 — soft에선 무시되고 hard/hybrid에서만 쓰인다.
    filter_domain = domains[0] if domains else None
    hits = await search_with_mode(qvec, top_k, domain=filter_domain, mode=effective_filter, settings=settings)

    results = [(point.score, point.payload or {}) for point in hits]

    # rerank 모드는 도메인 가산 전 base(rerank+최신성)를 만든 뒤 도메인 가산. dense는 vector score가 base.
    base = results if mode == "dense" else await _rerank_base(query, results, settings)
    ranked = _soft_ranked(base, domains, settings)

    top = ranked[:top_n]
    chunk_ids = [payload.get("chunk_id", "") for _, payload in top if payload.get("chunk_id")]
    top_score = top[0][0] if top else 0.0
    return RetrievalResult(chunk_ids=chunk_ids, top_score=top_score, n=len(chunk_ids))


async def ranked_chunk_ids(
    query: str,
    *,
    mode: Mode,
    domains: list[str] | None = None,
    filter_mode: FilterMode | None = None,
    top_k: int | None = None,
    top_n: int | None = None,
    settings: Settings | None = None,
) -> list[str]:
    """검색 지표용 — ranked chunk_id 리스트만 반환."""
    result = await retrieve_for_eval(
        query, mode=mode, domains=domains, filter_mode=filter_mode, top_k=top_k, top_n=top_n, settings=settings
    )
    return result.chunk_ids


def predicted_answerable(result: RetrievalResult, *, floor: float, min_docs: int) -> bool:
    """결정론 answerable 예측 — 1위 점수가 floor 이상이고 문서 수가 min_docs 이상.

    floor(τ)는 reranker 점수 분포에 의존 → 베이스라인 측정 후 보정한다.
    이 신호를 #B Trust 재검색 트리거가 재사용한다.
    """
    return result.n >= min_docs and result.top_score >= floor
