"""Trust Agent 노드 — 버전 계보 4축 채점 + collapse + 재검색 사다리 (#108 재설계).

규칙/메타데이터 기반(채점·판정은 LLM 미사용·결정론 — 쿼리 재작성만 LLM 1회).
retriever와 answer 사이에 위치한다.

처리 순서 (설계 문서 docs/Baemin/01_trust_agent_redesign.md v1.5):
    ① 재검색 진입이면 1차 생존 문서와 **합집합 병합** (chunk_id dedupe — 교체 금지)
    ② per-doc 채점: version_fit(currency/match 조건 분기) + per_doc_evidence
    ③ collapse: doc_key 그룹별 최적 1건 잔류 (비교 질의 target 버전은 면제)
    ④ coverage: 주제 단위 (비교 질의=회수율, strong-single-topic waiver)
    ⑤ overall = w·evidence_mean + w·coverage + w·(1−잔여dup)  — sensitivity는 게이트 전용
    ⑥ 재검색 사다리 (표 순서=우선순위): 재작성 / 버전 필터 / 주제 확장
    ⑦ 사다리 소진 후에만 게이트 판정 (deprecated_only / conflicting / sensitive_block)

상태 계약:
    읽기: documents, first_pass_documents, retry_count, max_retries, target_versions,
          refined_query, model
    쓰기: documents(collapse 생존), trust_score, gate_flags, should_re_retrieve,
          retry_action·excluded_doc_keys·pinned_doc_keys·version_filter(재검색 시),
          first_pass_documents, missing_versions, retry_count, refined_query(재작성 시),
          domains/domain(EXPAND_TOPICS 시 해제), agent_trace
"""

from __future__ import annotations

import logging
from collections import defaultdict
from functools import partial

import anyio

from app.agents.answer.answerability import GateFlags
from app.agents.retriever.rerank import recency_factor
from app.agents.state import AgentState, RetryAction, SourceDocument, TrustScore
from app.agents.trust.prompts import REWRITE_SYSTEM_PROMPT
from app.agents.trust.schema import RetryDecision, TrustOutput
from app.config import Settings, get_settings
from app.rag.labels import latest_version, versions_equal
from app.rag.lineage import get_lineages
from app.rag.version_fit import MATCH_MODE, compute_version_fit

logger = logging.getLogger(__name__)

# 구 5축 보고 계약의 중립 상수 (track-B 데이터 부재 — config가 아닌 모듈 상수로 강등)
_NEUTRAL = 1.0


# ---------------------------------------------------------------------------
# ① 병합 (재검색 결과는 교체가 아니라 합집합 — 설계 v1.4)
# ---------------------------------------------------------------------------


def merge_documents(first_pass: list[SourceDocument], new_docs: list[SourceDocument]) -> list[SourceDocument]:
    """1차 생존 문서 ∪ 재검색 결과. dedupe 키 = chunk_id (없으면 (page_id, hash)).

    제외 필터 재검색을 교체로 구현하면 원래의 강한 근거가 컨텍스트에서 통째로 사라진다.
    """
    merged: dict[tuple, SourceDocument] = {}
    for doc in [*first_pass, *new_docs]:
        key = (doc.chunk_id,) if doc.chunk_id else (doc.page_id, doc.hash)
        if key not in merged:
            merged[key] = doc
    return list(merged.values())


# ---------------------------------------------------------------------------
# ② per-doc 채점
# ---------------------------------------------------------------------------


def _group_key(doc: SourceDocument) -> str:
    """collapse/주제 집계 그룹 키. 계보 없는 문서(doc_key="")는 page 단위로 폴백."""
    return doc.doc_key or f"page:{doc.page_id}"


def annotate_version_fits(
    documents: list[SourceDocument],
    lineages: dict[str, frozenset[str]],
    target_versions: list[str],
    settings: Settings,
) -> None:
    """각 문서에 version_fit·per_doc_evidence를 기입한다 (in-place — dataclass 가변)."""
    for doc in documents:
        fit = compute_version_fit(
            product_version=doc.product_version,
            site=doc.site,
            eol=doc.is_eol,
            lineage=lineages.get(doc.doc_key, frozenset()),
            target_versions=target_versions,
            settings=settings,
        )
        authority = settings.site_tier.get(doc.site, settings.site_tier_default)
        doc.version_fit = fit.fit
        doc.version_fit_mode = fit.mode
        doc.raw_currency = fit.raw_currency
        # wsum 정규화 — env로 가중치 합이 1.0이 아니게 덮어써도 [0,1] 유지 (overall 블렌드와 동일 규약)
        wsum = settings.trust_w_version_fit + settings.trust_w_authority
        doc.per_doc_evidence = (settings.trust_w_version_fit * fit.fit + settings.trust_w_authority * authority) / (
            wsum or 1.0
        )


# ---------------------------------------------------------------------------
# ③ collapse — 버전 형제 정리 (설계 4.2: 점수보다 액션)
# ---------------------------------------------------------------------------


def _tiebreak_key(doc: SourceDocument, settings: Settings) -> tuple:
    """collapse 동률 타이브레이커 사다리 (설계 v1.3).

    version_fit → raw_currency(match 모드 전용 — currency 모드에선 fit과 동일해 무의미)
    → last_modified 반감기. ko/en 중복 같은 완전 동률은 마지막 단계에서 갈린다.
    """
    raw_currency = doc.raw_currency if doc.version_fit_mode == MATCH_MODE else 0.0
    recency = recency_factor(doc.last_modified, settings.rerank_recency_half_life_days)
    return (doc.version_fit, raw_currency, recency)


def collapse_siblings(
    documents: list[SourceDocument],
    target_versions: list[str],
    settings: Settings,
) -> tuple[list[SourceDocument], dict[str, list[str]]]:
    """doc_key 그룹별 최적 1건만 잔류시킨다.

    면제(설계 v1.4): target_versions가 복수(비교 질의)면 요청 버전과 일치하는 문서는
    버전당 1건씩 잔류 — "1.25→1.33 차이" 질의는 형제가 둘 다 필요하다.
    반환: (생존 문서, doc_key → 제거된 버전 목록)  — sources "다른 버전 N개" 표시용.
    """
    exempt = len(target_versions) >= 2
    groups: dict[str, list[SourceDocument]] = defaultdict(list)
    for doc in documents:
        groups[_group_key(doc)].append(doc)

    survivors: list[SourceDocument] = []
    removed_versions: dict[str, list[str]] = {}
    for key, docs in groups.items():
        keep: list[SourceDocument] = []
        rest = sorted(docs, key=partial(_tiebreak_key, settings=settings), reverse=True)
        if exempt:
            # 요청 버전당 최적 1건 면제 잔류 (해당 버전 내 동률도 타이브레이커 적용된 순서)
            for target in target_versions:
                for doc in rest:
                    if versions_equal(doc.product_version, target) and doc not in keep:
                        keep.append(doc)
                        break
        if not keep:  # 면제 대상 없음(또는 비교 질의 아님) → 그룹 최적 1건
            keep = [rest[0]]
        survivors.extend(keep)
        kept_ids = {id(d) for d in keep}
        removed = sorted({d.product_version for d in rest if id(d) not in kept_ids and d.product_version})
        if removed:
            removed_versions[key] = removed
    return survivors, removed_versions


# ---------------------------------------------------------------------------
# ④ coverage — 주제 충분성 (설계 4.4)
# ---------------------------------------------------------------------------


def compute_coverage(
    survivors: list[SourceDocument],
    target_versions: list[str],
    settings: Settings,
    rerank_fallback: bool = False,
) -> tuple[float, bool, list[str], int]:
    """반환: (coverage, waiver_applied, missing_versions, n_good_topics).

    비교 질의(복수 target)는 회수율 = 회수된 target 버전 수 / 요청 버전 수 (설계 v1.5) —
    min_topics 기준이면 한 버전만 회수돼도 통과해 비교 불가 컨텍스트로 답하는 오답 경로가 있다.
    그 외는 min(1, n_good_topics / trust_min_topics).

    rerank 정상: τ 비교는 **raw rerank 점수** 기준.
    rerank_fallback(리랭커 다운): raw가 전부 0이라 그 임계로는 못 거른다. 점수 스케일이 모드별로
    달라(dense cosine vs hybrid RRF) **절대 임계 대신 top 검색점수 대비 비율**(trust_fallback_score_ratio)
    로 good을 정한다 — 모드-독립. ratio=0이면 검색 생존 전부를 good으로 신뢰('측정 불가 ≠ 관련 없음').
    과신은 LLM 자기판정·인용 guard·version/authority 축이 견제한다.
    """
    if rerank_fallback:
        top_score = max((d.score for d in survivors), default=0.0)
        floor = settings.trust_fallback_score_ratio * top_score
        good = [d for d in survivors if d.score >= floor]
    else:
        good = [d for d in survivors if d.raw_rerank_score >= settings.trust_rerank_floor]
    n_good_topics = len({_group_key(d) for d in good})

    if len(target_versions) >= 2:
        retrieved = {t for t in target_versions if any(versions_equal(d.product_version, t) for d in survivors)}
        missing = [t for t in target_versions if t not in retrieved]
        return len(retrieved) / len(target_versions), False, missing, n_good_topics

    # strong-single-topic waiver (설계 4.4) — raw rerank 신뢰도 기반이라 폴백(rerank 부재)에선 미적용.
    if not rerank_fallback:
        tops = sorted((d.raw_rerank_score for d in survivors), reverse=True)
        if tops and tops[0] >= settings.trust_tau_strong:
            gap_ok = len(tops) == 1 or (tops[0] - tops[1]) >= settings.trust_gap_strong  # top2 부재 → 자동 충족
            if gap_ok:
                return 1.0, True, [], n_good_topics

    return min(1.0, n_good_topics / settings.trust_min_topics), False, [], n_good_topics


# ---------------------------------------------------------------------------
# ⑤ 잔여 중복 + overall
# ---------------------------------------------------------------------------


def residual_duplication(survivors: list[SourceDocument], target_versions: list[str]) -> float:
    """collapse 후 잔여 중복 — hash 성분 ∪ 형제 혼입 성분의 최대값.

    비교 질의로 면제된 형제 쌍(target 버전 일치 문서)은 형제 성분에서 제외한다(설계 v1.5) —
    설계가 스스로 명령한 동작("둘 다 남겨라")을 점수가 처벌하면 안 된다.
    """
    if not survivors:
        return 0.0
    hashes = [d.hash for d in survivors if d.hash]
    hash_dup = 1.0 - len(set(hashes)) / len(hashes) if hashes else 0.0

    countable = (
        [d for d in survivors if not any(versions_equal(d.product_version, t) for t in target_versions)]
        if len(target_versions) >= 2
        else survivors
    )
    if countable:
        pairs = {(_group_key(d), d.product_version) for d in countable}
        topics = {_group_key(d) for d in countable}
        sibling_dup = 1.0 - len(topics) / len(pairs) if pairs else 0.0
    else:
        sibling_dup = 0.0
    return max(hash_dup, sibling_dup)


def _sensitivity(documents: list[SourceDocument], cap: int) -> float:
    """[MASKED_*] 마커 밀도 → 민감정보 위험 [0,1]. 게이트 전용 (블렌드 제외, 설계 5.1)."""
    if cap <= 0 or not documents:
        return 0.0
    masked = sum(d.content_snippet.count("[MASKED_") for d in documents)
    return min(1.0, masked / cap)


def score_survivors(
    survivors: list[SourceDocument],
    target_versions: list[str],
    settings: Settings,
    rerank_fallback: bool = False,
) -> TrustOutput:
    """collapse 생존 집합을 4축으로 채점한다 (순수 함수 — 게이트는 evaluate_gates 별도)."""
    if not survivors:
        return TrustOutput(
            version_fit_mean=0.0, coverage=0.0, residual_duplication=0.0, authority_mean=0.0, overall=0.0
        )
    coverage, waiver, missing, n_good_topics = compute_coverage(
        survivors, target_versions, settings, rerank_fallback=rerank_fallback
    )
    dup = residual_duplication(survivors, target_versions)
    evidence_mean = sum(d.per_doc_evidence for d in survivors) / len(survivors)
    fit_mean = sum(d.version_fit for d in survivors) / len(survivors)
    authority_mean = sum(settings.site_tier.get(d.site, settings.site_tier_default) for d in survivors) / len(survivors)

    overall = (
        settings.trust_w_evidence_mean * evidence_mean
        + settings.trust_w_coverage * coverage
        + settings.trust_w_residual_dup * (1.0 - dup)
    )
    wsum = settings.trust_w_evidence_mean + settings.trust_w_coverage + settings.trust_w_residual_dup
    overall = max(0.0, min(1.0, overall / (wsum or 1.0)))

    recency = max(recency_factor(d.last_modified, settings.rerank_recency_half_life_days) for d in survivors)
    return TrustOutput(
        version_fit_mean=fit_mean,
        coverage=coverage,
        residual_duplication=dup,
        authority_mean=authority_mean,
        overall=overall,
        waiver_applied=waiver,
        n_good_topics=n_good_topics,
        recency=recency,
        owner_trust=_NEUTRAL,
        verification_label=_NEUTRAL,
        sensitivity_risk=_sensitivity(survivors, settings.trust_sensitivity_masked_cap),
    )


# ---------------------------------------------------------------------------
# ⑥ 재검색 사다리 (설계 6장 — 표 순서 = 우선순위)
# ---------------------------------------------------------------------------


def decide_retry_action(
    survivors: list[SourceDocument],
    target_versions: list[str],
    lineages: dict[str, frozenset[str]],
    output: TrustOutput,
    missing_versions: list[str],
    settings: Settings,
) -> RetryDecision:
    """실패 원인 진단 → 전략 선택. 행 우선순위는 선언 순서(설계 v1.5).

    호출 전제: retry 한도 미소진 (한도 체크는 trust_node가 최우선으로 수행).
    """
    # 1) 관련 근거 전무 → 쿼리 재작성 (실패 원인이 쿼리 자체일 확률 최대)
    if output.n_good_topics == 0:
        return RetryDecision(action=RetryAction.REWRITE_QUERY)

    # 2) [match] 미회수 target 버전이 계보에 존재 → 해당 버전 필터 재검색
    #    계보에도 없으면(미색인) 행 생략 — 빈손 재검색 대신 coverage 결손(회수율)으로 진행
    for missing in missing_versions:
        for doc in survivors:
            lineage = lineages.get(doc.doc_key, frozenset())
            indexed = next((v for v in lineage if versions_equal(v, missing)), "")
            if indexed:
                pinned = sorted(
                    {d.doc_key for d in survivors if d.doc_key and indexed in lineages.get(d.doc_key, frozenset())}
                )
                return RetryDecision(action=RetryAction.RETRY_VERSION, version_filter=indexed, pinned_doc_keys=pinned)

    # 3) [currency] 생존 전부 EOL & 계보에 더 새 버전 존재 → latest 필터 재검색
    #    계보에 새 버전 없으면 PROCEED (빈손 재검색 금지 — 게이트에서 OUTDATED 직행, 설계 v1.4)
    if not target_versions and survivors and all(d.is_eol for d in survivors):
        # 계보별 최신 버전이 서로 다를 수 있다(예: apache 2.4 vs k8s v1.33 혼재) —
        # 단일 version_filter에 이질 doc_key를 묶으면 일부 주제가 영원히 미회수된다.
        # 최신 버전값으로 그룹핑해 가장 많은 doc_key를 구제하는 그룹 하나만 재검색한다.
        pinned_by_latest: dict[str, set[str]] = defaultdict(set)
        for doc in survivors:
            lineage = lineages.get(doc.doc_key, frozenset())
            latest = latest_version(lineage)
            if doc.doc_key and latest and not versions_equal(latest, doc.product_version):
                pinned_by_latest[latest].add(doc.doc_key)
        if pinned_by_latest:
            newer = max(pinned_by_latest, key=lambda v: (len(pinned_by_latest[v]), v))
            return RetryDecision(
                action=RetryAction.RETRY_VERSION,
                version_filter=newer,
                pinned_doc_keys=sorted(pinned_by_latest[newer]),
            )
        return RetryDecision(action=RetryAction.PROCEED)

    # 4) 주제 부족 (waiver·비교 질의 제외) → 확보 doc_key 제외하고 폭 확대
    if not output.waiver_applied and len(target_versions) < 2 and 0 < output.n_good_topics < settings.trust_min_topics:
        return RetryDecision(
            action=RetryAction.EXPAND_TOPICS,
            excluded_doc_keys=sorted({d.doc_key for d in survivors if d.doc_key}),
        )

    return RetryDecision(action=RetryAction.PROCEED)


async def _rewrite_query(refined_query: str, model: str, settings: Settings) -> str:
    """LLM 쿼리 재작성. 실패하면 원 쿼리 유지 (재작성 없이 PROCEED 경로로)."""
    from app.services.llm_selector import call_llm  # 순환 import 회피 — 사용 시점 로드

    try:
        rewritten = await call_llm(REWRITE_SYSTEM_PROMPT, refined_query, model=model or settings.trust_rewrite_model)
        rewritten = rewritten.strip().strip('"')
        return rewritten or refined_query
    except Exception:
        logger.warning("쿼리 재작성 LLM 실패 — 원 쿼리 유지", exc_info=True)
        return refined_query


# ---------------------------------------------------------------------------
# ⑦ 게이트 — 사다리 소진 후 최종 진입에서만 판정 (설계 v1.5)
# ---------------------------------------------------------------------------


def evaluate_gates(survivors: list[SourceDocument], sensitivity: float, settings: Settings) -> GateFlags:
    """게이트가 사다리보다 먼저 판정되면 retry 기회 없이 OUTDATED 직행하는 사고가 난다 —
    반드시 사다리 소진(PROCEED 확정) 후에 호출한다.
    """
    deprecated_only = bool(survivors) and all(d.is_eol for d in survivors)
    return GateFlags(
        conflicting=_conflicting(survivors, settings),
        deprecated_only=deprecated_only,
        sensitive_block=sensitivity >= 1.0,
    )


def _conflicting(survivors: list[SourceDocument], settings: Settings) -> bool:
    """동등 권위 충돌 의심: **같은 site·같은 product_version**의 서로 다른 주제가
    둘 다 raw ≥ floor이고 점수 격차 < gap (설계 v1.3 site 한정 + v1.5 버전 한정).

    product_version 상이 쌍 제외 — doc_key 정규화 실패로 collapse를 우회한 미결합 형제
    (워크스루 D: 고아 2.2 vs 정상 2.4)가 충돌로 오탐되는 두 번째 출구를 차단한다.
    버전 층위가 다르면 충돌 혐의보다 미결합 형제 혐의가 우선이다.
    """
    floor, gap = settings.trust_rerank_floor, settings.trust_conflict_score_gap
    by_topic: dict[tuple[str, str, str], float] = {}
    for d in survivors:
        bucket = (d.site, d.product_version, _group_key(d))
        by_topic[bucket] = max(by_topic.get(bucket, d.raw_rerank_score), d.raw_rerank_score)
    # 같은 (site, version) 층위 안에서 서로 다른 주제끼리만 비교
    by_tier: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (site, version, _key), top in by_topic.items():
        by_tier[(site, version)].append(top)
    for (site, _version), tops in by_tier.items():
        if not site or len(tops) < 2:
            continue
        ordered = sorted(tops, reverse=True)
        if ordered[1] >= floor and (ordered[0] - ordered[1]) < gap:
            return True
    return False


# ---------------------------------------------------------------------------
# 노드
# ---------------------------------------------------------------------------


async def trust_node(state: AgentState) -> dict:
    """Strategy별 Trust 계약을 dispatch한다."""
    if state.get("retriever_strategy") == "single_agentic":
        return await evaluate_trust_node(state)
    return await deterministic_trust_node(state)


async def evaluate_trust_node(state: AgentState) -> dict:
    """Single Agentic 경로용 rules-only evaluator. 재작성·routing 신호를 만들지 않는다."""
    retry = state.get("retry_count", 0)
    evaluated = await deterministic_trust_node(
        {
            **state,
            "first_pass_documents": [],
            "retry_count": retry,
            "max_retries": retry,
        }
    )
    return {
        key: evaluated[key]
        for key in ("documents", "trust_score", "gate_flags", "missing_versions", "agent_trace")
        if key in evaluated
    }


async def deterministic_trust_node(state: AgentState) -> dict:
    """문서를 채점하고 재검색 사다리를 결정한다."""
    settings = get_settings()
    documents = state.get("documents", [])
    first_pass = state.get("first_pass_documents", [])
    retry = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.trust_max_retries)
    target_versions = [str(v) for v in state.get("target_versions", [])]
    # 리랭커 폴백 여부 — coverage 산정 시 raw rerank τ 대신 검색점수 비율을 쓰게 한다 (#202)
    rerank_fallback = bool(state.get("rerank_fallback", False))

    # ① 재검색 진입이면 합집합 병합 (교체 금지 — 설계 v1.4)
    if first_pass:
        documents = merge_documents(first_pass, documents)

    # ② 계보 조회 + per-doc 채점
    doc_keys = [d.doc_key for d in documents]
    lineages = await anyio.to_thread.run_sync(partial(get_lineages, doc_keys, settings=settings))
    annotate_version_fits(documents, lineages, target_versions, settings)

    # ③ collapse → ④⑤ 채점
    survivors, _removed = collapse_siblings(documents, target_versions, settings) if documents else ([], {})
    output = score_survivors(survivors, target_versions, settings, rerank_fallback=rerank_fallback)
    if survivors:
        _coverage, _waiver, missing_versions, _n = compute_coverage(
            survivors, target_versions, settings, rerank_fallback=rerank_fallback
        )
    else:  # 빈 결과: 비교 질의면 전 버전 미회수, 아니면 missing 개념 없음
        missing_versions = list(target_versions) if len(target_versions) >= 2 else []

    result: dict = {
        "documents": survivors,  # collapse 생존이 Answer 컨텍스트 (의도된 축소)
        "trust_score": TrustScore(
            recency=output.recency,
            verification_label=output.verification_label,
            owner_trust=output.owner_trust,
            duplication_conflict=output.residual_duplication,
            sensitivity_risk=output.sensitivity_risk,
            overall=output.overall,
            version_fit_mean=output.version_fit_mean,
            coverage=output.coverage,
            residual_duplication=output.residual_duplication,
            authority_mean=output.authority_mean,
            waiver_applied=output.waiver_applied,
        ),
        "missing_versions": missing_versions,
        "agent_trace": ["trust"],
    }

    # ⑥ 사다리 — 한도 체크 최우선 (소진 시 진단 없이 PROCEED)
    decision = RetryDecision(action=RetryAction.PROCEED)
    if retry < max_retries and documents:
        decision = decide_retry_action(survivors, target_versions, lineages, output, missing_versions, settings)
    elif retry < max_retries and not documents:
        decision = RetryDecision(action=RetryAction.REWRITE_QUERY)

    if decision.action != RetryAction.PROCEED:
        result["should_re_retrieve"] = True
        result["retry_action"] = decision.action
        result["retry_count"] = retry + 1
        result["first_pass_documents"] = survivors  # 다음 진입에서 병합할 1차 생존
        result["excluded_doc_keys"] = decision.excluded_doc_keys
        result["pinned_doc_keys"] = decision.pinned_doc_keys
        result["version_filter"] = decision.version_filter
        if decision.action == RetryAction.REWRITE_QUERY:
            # 재작성은 검색 텍스트만 변형 — target_versions·domains는 불변 (모드 보존, 설계 v1.5)
            result["refined_query"] = await _rewrite_query(
                state.get("refined_query", ""), state.get("model", ""), settings
            )
        if decision.action == RetryAction.EXPAND_TOPICS:
            result["domains"] = []  # 도메인 가산 해제로 폭 확대
            result["domain"] = None  # 하위호환 파생값 동기화
        return result

    # ⑦ PROCEED 확정 → 게이트 판정 (사다리 소진 후에만 — 설계 v1.5)
    gates = evaluate_gates(survivors, output.sensitivity_risk, settings)
    result["should_re_retrieve"] = False
    result["retry_action"] = RetryAction.PROCEED
    result["gate_flags"] = gates
    return result


def trust_decision(state: AgentState) -> str:
    """근거 부족 시 재검색(retriever), 충분하면 answer로 분기 (graph.py 계약 유지)."""
    return "retriever" if state.get("should_re_retrieve") else "answer"
