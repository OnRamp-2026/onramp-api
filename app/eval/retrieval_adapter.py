"""검색 평가 어댑터 — query를 우리 검색 경로로 흘려 ranked chunk_id를 반환한다.

`retrieve_node`(app/agents/retriever/node.py)와 같은 검색 코어(search_with_mode)를 써서
SourceDocument 대신 평가용 chunk_id 리스트/점수를 돌려준다.
LLM-free(임베딩 검색만) → Router를 거치지 않고 raw query를 그대로 검색에 투입한다.

mode (검색 방식):
    "dense"  — 순수 Qdrant kNN (vector + 도메인 가산). **HYBRID_SEARCH_ENABLED 무관**(플래그 독립 진단).
    "sparse" — OpenSearch BM25 단독 (lexical). 임베딩 불필요.
    "hybrid" — Dense+BM25 RRF 융합 (플래그 무관·명시적). 리랭커 없음(1차 검색).
    "rerank" — 운영 경로 미러: 1차 검색(HYBRID_SEARCH_ENABLED면 hybrid) + Cross-Encoder 재정렬 +
               최신성·도메인·버전·권위 부스트 체인. **게이트 대상 지표**.
filter_mode (도메인 필터 전략, None이면 config 기본=운영과 동일):
    "hard" / "hybrid" / "soft" — search_with_mode 참고
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import anyio

from app.agents.retriever.rerank import (
    apply_domain_weight,
    apply_metadata_weight,
    apply_ranking_boosts,
    get_reranker,
)
from app.agents.retriever.search import FilterMode, dense_search, search_with_mode
from app.config import Settings, get_settings
from app.rag.embedder import get_embedder
from app.rag.lineage import get_lineages

logger = logging.getLogger(__name__)

Mode = Literal["dense", "sparse", "hybrid", "rerank"]


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


def _first_stage_result(
    results: list[tuple[float, dict]], domains: list[str] | None, settings: Settings, top_n: int
) -> RetrievalResult:
    """1차 검색(dense/sparse/hybrid) 공통 — 도메인 soft 가산 정렬 후 top_n RetrievalResult.

    리랭커가 없으므로 raw 점수 분리가 없다 → tau_score=top_score(1차 정렬 점수).
    """
    ranked = _soft_ranked(results, domains, settings)[:top_n]
    chunk_ids = [payload.get("chunk_id", "") for _, payload in ranked if payload.get("chunk_id")]
    top_score = ranked[0][0] if ranked else 0.0
    return RetrievalResult(chunk_ids=chunk_ids, top_score=top_score, n=len(chunk_ids), tau_score=top_score)


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
    top_score: float  # 1위 정렬 점수 (rerank 모드=부스트 합산 ranking / dense 모드=vector+도메인 가산)
    n: int  # 반환된 문서 수
    # 점수 분리 (#103): τ 진단은 raw 기준. dense 모드는 raw 부재 → tau_score=top_score(vector).
    tau_score: float = 0.0  # answerability/재검색 τ와 비교할 점수 (rerank 모드=top_n 내 최대 raw)
    raw_scores: tuple[float, ...] = field(default_factory=tuple)  # top_n raw, 순위(ranking) 순 — τ_strong/gap 보정용


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
    """query를 검색해 ranked chunk_id와 점수를 반환한다 (retrieve_node와 동일 코어).

    rerank 모드는 운영 경로와 동일한 부스트 체인(최신성·도메인·버전·권위, #103)으로 정렬하되
    raw 점수를 분리 보존한다. dense 모드는 진단용 baseline — vector+도메인 가산만(역사적 비교 유지).
    domains: 질의 도메인 집합(순서 우선). soft 가산이 문서 단일 domain을 이 집합과 비교한다.
    """
    settings = settings or get_settings()
    top_k = settings.retriever_top_k if top_k is None else top_k
    top_n = settings.retriever_top_n if top_n is None else top_n
    if top_k <= 0 or top_n <= 0:
        raise ValueError(f"top_k/top_n 은 1 이상이어야 합니다: top_k={top_k}, top_n={top_n}")
    effective_filter = filter_mode if filter_mode is not None else settings.retriever_domain_filter_mode
    # 필터용 domain은 대표(domains[0])만 — soft에선 무시되고 hard/hybrid에서만 쓰인다.
    filter_domain = domains[0] if domains else None
    # 명시적 단일 방식(sparse/hybrid) 진단 — soft면 하드필터 없음(운영 soft와 동일), hard/hybrid면 도메인 필터.
    method_domain = None if effective_filter == "soft" else filter_domain

    # --- sparse: BM25 단독 (임베딩 불필요) ---
    if mode == "sparse":
        from app.db.opensearch import get_opensearch

        bm25 = await get_opensearch().search(
            query, top_k=top_k, tenant_id=settings.auth_default_tenant, domain=method_domain
        )
        results = [(hit.score, hit.payload or {}) for hit in bm25]
        return _first_stage_result(results, domains, settings, top_n)

    qvec = await get_embedder().embed_query(query)

    # --- hybrid: Dense+BM25 RRF (플래그 무관·명시적 융합, 리랭커 없음) ---
    if mode == "hybrid":
        from app.rag.hybrid_search import hybrid_search

        hits = await hybrid_search(
            query,
            qvec,
            top_k=top_k,
            domain=method_domain,
            filters=None,
            settings=settings,
            dense_search_fn=dense_search,
        )
        results = [(point.score, point.payload or {}) for point in hits]
        return _first_stage_result(results, domains, settings, top_n)

    # --- dense: 순수 Qdrant kNN (query_text="" → hybrid 게이트 우회, HYBRID_SEARCH_ENABLED 무관) ---
    if mode == "dense":
        hits = await search_with_mode(
            qvec, top_k, domain=filter_domain, mode=effective_filter, query_text="", settings=settings
        )
        results = [(point.score, point.payload or {}) for point in hits]
        return _first_stage_result(results, domains, settings, top_n)

    # --- rerank: 운영 경로 미러 — 1차 검색(플래그면 hybrid) → raw 리랭킹 → 계보 조회 → 부스트 체인 정렬 ---
    hits = await search_with_mode(
        qvec, top_k, domain=filter_domain, mode=effective_filter, query_text=query, settings=settings
    )
    results = [(point.score, point.payload or {}) for point in hits]
    doc_keys = [payload.get("doc_key", "") or "" for _, payload in results]
    lineages = await anyio.to_thread.run_sync(partial(get_lineages, doc_keys, settings=settings))
    candidates = [(payload.get("content", ""), payload) for _, payload in results]
    try:
        reranked = await anyio.to_thread.run_sync(get_reranker().rerank, query, candidates)
        rows = [
            (apply_ranking_boosts(raw, payload, domains, lineages, [], settings), raw, payload)
            for raw, payload in reranked
        ]
    except Exception as exc:  # 리랭커 실패 → vector 폴백 (운영 _vector_fallback 동일: raw=0.0)
        logger.warning("리랭커 실패로 dense 폴백: %s", exc, exc_info=True)
        rows = [
            (apply_ranking_boosts(vec, payload, domains, lineages, [], settings), 0.0, payload)
            for vec, payload in results
        ]
    rows.sort(key=lambda item: item[0], reverse=True)

    top = rows[:top_n]
    chunk_ids = [payload.get("chunk_id", "") for _, _, payload in top if payload.get("chunk_id")]
    raw_scores = tuple(raw for _, raw, _ in top)
    return RetrievalResult(
        chunk_ids=chunk_ids,
        top_score=top[0][0] if top else 0.0,
        n=len(chunk_ids),
        tau_score=max(raw_scores, default=0.0),  # Trust should_re_retrieve와 동일: top_n 내 최대 raw
        raw_scores=raw_scores,
    )


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
    """결정론 answerable 예측 — τ 비교 점수(tau_score)가 floor 이상이고 문서 수가 min_docs 이상.

    tau_score는 rerank 모드에선 raw 점수(#103 점수 분리 — Trust should_re_retrieve와 동일 신호),
    dense 모드에선 vector top 점수. floor(τ)는 점수 분포에 의존 → calibrate_answerability.py로 보정.
    """
    return result.n >= min_docs and result.tau_score >= floor
