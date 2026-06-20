"""Trust Agent 단위 테스트 (#108 재설계 — 규칙기반·결정론, Qdrant/LLM monkeypatch).

설계 문서 11장 워크스루 A–E를 1:1 회귀 테스트로 고정한다 — 문서가 "규칙 변경 시
이 표를 다시 따라가라"고 선언한 것을 코드로 옮긴 것. 교차 상호작용 버그(v1.4·v1.5에서
발견된 종류)는 축별 명세가 아니라 이런 end-to-end 케이스에서만 잡힌다.
"""

from datetime import UTC, datetime, timedelta

from app.agents.state import RetryAction, SourceDocument, TrustScore
from app.agents.trust import node as node_mod
from app.agents.trust.node import (
    collapse_siblings,
    compute_coverage,
    decide_retry_action,
    merge_documents,
    residual_duplication,
    score_survivors,
    trust_decision,
    trust_node,
)
from app.config import Settings

_RECENT = datetime.now(UTC).isoformat()
_OLD = (datetime.now(UTC) - timedelta(days=3000)).isoformat()

S = Settings()
_ABOVE = S.trust_rerank_floor + 0.05  # τ 위 (waiver τ_strong 0.90 아래일 수도 — 케이스별 명시)

# 워크스루 공용 계보
K8S_LINEAGE = frozenset({"v1.18", "v1.25", "v1.29", "v1.33"})


def _doc(
    raw=0.9,
    page_id="p1",
    doc_key="",
    site="",
    version="",
    eol=False,
    h="",
    chunk_id="",
    last_modified=_RECENT,
    content="내용",
) -> SourceDocument:
    return SourceDocument(
        title="t",
        content_snippet=content,
        rerank_score=raw + 0.1,  # ranking은 부스트 합산 — trust는 raw만 본다
        raw_rerank_score=raw,
        page_id=page_id,
        last_modified=last_modified,
        hash=h or f"h-{page_id}-{version}",
        chunk_id=chunk_id or f"c-{page_id}-{version}",
        site=site,
        product_version=version,
        doc_key=doc_key,
        is_eol=eol,
    )


def _patch_lineages(monkeypatch, lineages: dict):
    monkeypatch.setattr(node_mod, "get_lineages", lambda keys, **kw: {k: lineages.get(k, frozenset()) for k in keys})


async def _run(monkeypatch, docs, *, lineages=None, target=None, retry=0, max_retries=1, **extra):
    _patch_lineages(monkeypatch, lineages or {})
    state = {
        "documents": docs,
        "retry_count": retry,
        "max_retries": max_retries,
        "target_versions": target or [],
        "refined_query": "q",
        **extra,
    }
    return await trust_node(state)


# ---------------------------------------------------------------------------
# 워크스루 A — 단일 사실 질의: collapse 후 waiver로 재검색 생략
# ---------------------------------------------------------------------------


async def test_walkthrough_a_single_fact_waiver(monkeypatch) -> None:
    docs = [
        _doc(raw=0.95, page_id="flag-133", doc_key="k8s:flag", site="kubernetes", version="v1.33"),
        _doc(raw=0.90, page_id="flag-129", doc_key="k8s:flag", site="kubernetes", version="v1.29"),
        _doc(raw=0.30, page_id="other", doc_key="k8s:other", site="kubernetes", version="v1.33"),
    ]
    lineages = {"k8s:flag": K8S_LINEAGE, "k8s:other": frozenset({"v1.33"})}
    out = await _run(monkeypatch, docs, lineages=lineages, target=["1.33"])

    survivors = out["documents"]
    flag_docs = [d for d in survivors if d.doc_key == "k8s:flag"]
    assert len(flag_docs) == 1 and flag_docs[0].product_version == "v1.33"  # 형제 collapse
    assert flag_docs[0].version_fit == 1.0 and flag_docs[0].version_fit_mode == "match"
    # waiver: collapse 후 top1 raw 0.95 ≥ τ_strong, 격차 0.95−0.30 ≥ gap_strong → coverage 1.0
    assert out["trust_score"].waiver_applied is True
    assert out["trust_score"].coverage == 1.0
    assert out["retry_action"] == RetryAction.PROCEED
    assert out["should_re_retrieve"] is False
    assert out["gate_flags"].deprecated_only is False and out["gate_flags"].conflicting is False


# ---------------------------------------------------------------------------
# 워크스루 B — 버전 비교 질의: 면제 잔류 + 회수율 + dup 면제
# ---------------------------------------------------------------------------


async def test_walkthrough_b_comparison_exempt(monkeypatch) -> None:
    docs = [
        _doc(raw=0.88, page_id="up-125", doc_key="k8s:upgrade", site="kubernetes", version="v1.25"),
        _doc(raw=0.88, page_id="up-133", doc_key="k8s:upgrade", site="kubernetes", version="v1.33"),
    ]
    out = await _run(monkeypatch, docs, lineages={"k8s:upgrade": K8S_LINEAGE}, target=["1.25", "1.33"])

    assert len(out["documents"]) == 2  # 면제 — 둘 다 잔류 (비교 질의는 형제가 둘 다 필요)
    assert out["trust_score"].coverage == 1.0  # 회수율 2/2
    assert out["trust_score"].residual_duplication == 0.0  # 면제 쌍 dup 제외 (v1.5)
    assert out["missing_versions"] == []
    assert out["retry_action"] == RetryAction.PROCEED


# ---------------------------------------------------------------------------
# 워크스루 C — EOL-only: 계보 사전 확인 → 버전 필터 재검색 / 게이트 직행
# ---------------------------------------------------------------------------


async def test_walkthrough_c1_eol_with_newer_in_lineage_retries(monkeypatch) -> None:
    docs = [
        _doc(raw=0.95, page_id="mpm-a", doc_key="apache:mpm", site="apache", version="2.2", eol=True),
        _doc(raw=0.92, page_id="mpm-b", doc_key="apache:mpm", site="apache", version="2.2", eol=True),
    ]
    out = await _run(monkeypatch, docs, lineages={"apache:mpm": frozenset({"2.2", "2.4"})})

    assert out["retry_action"] == RetryAction.RETRY_VERSION
    assert out["version_filter"] == "2.4"  # 계보에 실존하는 더 새 버전만 필터
    assert out["pinned_doc_keys"] == ["apache:mpm"]
    assert out["should_re_retrieve"] is True
    assert out["first_pass_documents"]  # 병합용 1차 생존 보존
    assert "gate_flags" not in out  # 게이트는 사다리 소진 후에만 (v1.5)


async def test_walkthrough_c2_eol_without_newer_proceeds_to_gate(monkeypatch) -> None:
    docs = [_doc(raw=0.95, page_id="mpm", doc_key="apache:mpm", site="apache", version="2.2", eol=True)]
    out = await _run(monkeypatch, docs, lineages={"apache:mpm": frozenset({"2.2"})})

    # 계보에 새 버전 없음 → 빈손 재검색 금지, 즉시 PROCEED + OUTDATED 게이트 (v1.4)
    assert out["retry_action"] == RetryAction.PROCEED
    assert out["gate_flags"].deprecated_only is True


# ---------------------------------------------------------------------------
# 워크스루 D — 정규화 실패 고아: 충돌 게이트 미발동 + EOL 캡 순서
# ---------------------------------------------------------------------------


async def test_walkthrough_d_orphan_no_conflict(monkeypatch) -> None:
    docs = [
        _doc(raw=0.90, page_id="orph", doc_key="apache:proxy-orphan", site="apache", version="2.2", eol=True),
        _doc(raw=0.89, page_id="norm", doc_key="apache:proxy", site="apache", version="2.4"),
    ]
    lineages = {
        "apache:proxy-orphan": frozenset({"2.2"}),  # 고아 — 단일 계보로 보임
        "apache:proxy": frozenset({"2.2", "2.4"}),
    }
    out = await _run(monkeypatch, docs, lineages=lineages)

    # 5.2 조건(같은 site·다른 doc_key·격차 0.01<gap)을 채우지만 product_version 상이 쌍 제외(v1.5)
    assert out["gate_flags"].conflicting is False
    orphan = next(d for d in out["documents"] if d.doc_key == "apache:proxy-orphan")
    # EOL 캡은 모든 분기의 최종 출력에 min(v1.4): 단일계보 보수캡 0.7 → 0.3
    assert orphan.version_fit == S.trust_eol_cap
    assert out["retry_action"] == RetryAction.PROCEED  # 주제 2개 충족 (알려진 과대 집계 — 9.1)


async def test_score_tie_does_not_block_complementary_documents(monkeypatch) -> None:
    """점수 동률만으로 실제 모순이 없는 보완 문서를 차단하지 않는다."""
    docs = [
        _doc(
            raw=0.931,
            page_id="a",
            doc_key="datadog:stdout",
            site="datadog",
            version="latest",
            content="컨테이너 애플리케이션은 STDOUT으로 로그를 출력한다.",
        ),
        _doc(
            raw=0.930,
            page_id="b",
            doc_key="datadog:stderr",
            site="datadog",
            version="latest",
            content="오류 로그는 STDERR로 출력한다.",
        ),
    ]
    out = await _run(monkeypatch, docs, lineages={})
    assert out["gate_flags"].conflicting is False


async def test_masked_secret_configuration_query_is_not_blocked(monkeypatch) -> None:
    docs = [
        _doc(
            raw=0.95,
            page_id="secret-config",
            content=(
                "[MASKED_SECRET] [MASKED_TOKEN] [MASKED_PASSWORD] "
                "[MASKED_KEY] [MASKED_CREDENTIAL]을 SealedSecret으로 생성하고 ArgoCD에 배포한다."
            ),
        )
    ]
    out = await _run(
        monkeypatch,
        docs,
        retry=1,
        max_retries=1,
        query="ArgoCD credential bootstrap의 목적과 Secret 구성을 설명해 주세요.",
    )
    assert out["trust_score"].sensitivity_risk == 1.0
    assert out["gate_flags"].sensitive_block is False


async def test_masked_secret_value_request_is_blocked(monkeypatch) -> None:
    docs = [
        _doc(
            raw=0.95,
            page_id="secret-value",
            content="[MASKED_SECRET] [MASKED_TOKEN] [MASKED_PASSWORD] [MASKED_KEY] [MASKED_CREDENTIAL]",
        )
    ]
    out = await _run(
        monkeypatch,
        docs,
        retry=1,
        max_retries=1,
        query="실제 GitHub token 값과 비밀번호를 알려주세요.",
    )
    assert out["gate_flags"].sensitive_block is True


# ---------------------------------------------------------------------------
# 워크스루 E — 비교 질의, 한 버전만 회수: 버전 필터 재검색 → 실패 시 회수율 결손
# ---------------------------------------------------------------------------


async def test_walkthrough_e_missing_version_retries(monkeypatch) -> None:
    docs = [_doc(raw=0.88, page_id="up-133", doc_key="k8s:upgrade", site="kubernetes", version="v1.33")]
    out = await _run(
        monkeypatch, docs, lineages={"k8s:upgrade": frozenset({"v1.25", "v1.33"})}, target=["1.25", "1.33"]
    )

    assert out["retry_action"] == RetryAction.RETRY_VERSION
    assert out["version_filter"] == "v1.25"  # 계보의 payload 표기값 (질의 표기 "1.25"와 동치 매칭)
    assert out["pinned_doc_keys"] == ["k8s:upgrade"]


async def test_walkthrough_e2_exhausted_reports_missing(monkeypatch) -> None:
    docs = [_doc(raw=0.88, page_id="up-133", doc_key="k8s:upgrade", site="kubernetes", version="v1.33")]
    out = await _run(
        monkeypatch,
        docs,
        lineages={"k8s:upgrade": frozenset({"v1.25", "v1.33"})},
        target=["1.25", "1.33"],
        retry=1,  # 한도 소진 — 진단 없이 PROCEED
    )

    assert out["retry_action"] == RetryAction.PROCEED
    assert out["trust_score"].coverage == 0.5  # 회수율 1/2 — 결손이 점수에 드러남 (v1.5)
    assert out["missing_versions"] == ["1.25"]  # Answer가 PARTIALLY 사유로 사용


# ---------------------------------------------------------------------------
# 재검색 사다리 — 우선순위·재작성·병합
# ---------------------------------------------------------------------------


async def test_ladder_rewrite_when_no_good_topics(monkeypatch) -> None:
    async def _fake_llm(system, user, **kw):
        return "재작성된 검색 쿼리"

    monkeypatch.setattr("app.services.llm_selector.call_llm", _fake_llm)
    docs = [_doc(raw=0.10, page_id="weak")]  # 전부 τ 미달 → 관련 근거 전무
    out = await _run(monkeypatch, docs, target=["1.25"])

    assert out["retry_action"] == RetryAction.REWRITE_QUERY
    assert out["refined_query"] == "재작성된 검색 쿼리"
    assert "target_versions" not in out  # 1차 추출값 고정 — trust는 건드리지 않는다 (모드 보존)


async def test_ladder_rewrite_llm_failure_keeps_query(monkeypatch) -> None:
    async def _boom(system, user, **kw):
        raise RuntimeError("LLM down")

    monkeypatch.setattr("app.services.llm_selector.call_llm", _boom)
    out = await _run(monkeypatch, [_doc(raw=0.10)])
    assert out["retry_action"] == RetryAction.REWRITE_QUERY
    assert out["refined_query"] == "q"  # 원 쿼리 유지


async def test_ladder_priority_version_before_topics(monkeypatch) -> None:
    """전부 옛 버전 + 주제 미달이 동시 성립하면 상위 행(RETRY_VERSION)이 우선 (v1.5)."""
    docs = [_doc(raw=0.95, page_id="old", doc_key="apache:mpm", site="apache", version="2.2", eol=True)]
    out = await _run(monkeypatch, docs, lineages={"apache:mpm": frozenset({"2.2", "2.4"})})
    assert out["retry_action"] == RetryAction.RETRY_VERSION  # EXPAND_TOPICS 아님


async def test_ladder_expand_topics_excludes_known_keys(monkeypatch) -> None:
    docs = [_doc(raw=0.95, page_id="only", doc_key="apache:only", site="apache", version="2.4")]
    # waiver 미발동을 위해 τ_strong 위지만 격차... 생존 1건이면 top2 부재 → waiver 자동 충족이라
    # raw를 τ_strong(0.90) 아래·τ(0.5229) 위로 둔다 → waiver 미발동 + 주제 1개 < 2
    docs = [_doc(raw=0.88, page_id="only", doc_key="apache:only", site="apache", version="2.4")]
    out = await _run(monkeypatch, docs, lineages={"apache:only": frozenset({"2.4"})})

    assert out["retry_action"] == RetryAction.EXPAND_TOPICS
    assert out["excluded_doc_keys"] == ["apache:only"]  # 확보 주제 제외 — 새 주제 발견 목적
    assert out["domains"] == [] and out["domain"] is None  # 도메인 가산 해제


async def test_merge_on_retry_is_union_not_replace(monkeypatch) -> None:
    """재검색 진입 시 1차 생존 ∪ 재검색 결과 — 교체 금지 (v1.4)."""
    first = _doc(raw=0.92, page_id="strong", doc_key="apache:strong", site="apache", version="2.4", chunk_id="c1")
    new = _doc(raw=0.90, page_id="fresh", doc_key="apache:fresh", site="apache", version="2.4", chunk_id="c2")
    out = await _run(
        monkeypatch,
        [new],
        retry=1,  # 한도 소진 → PROCEED (병합 결과 검증에 집중)
        first_pass_documents=[first],
    )
    keys = {d.doc_key for d in out["documents"]}
    assert keys == {"apache:strong", "apache:fresh"}  # 원래의 강한 근거가 살아 있다


def test_merge_documents_dedupes_by_chunk_id() -> None:
    a = _doc(page_id="p", chunk_id="c1")
    b = _doc(page_id="p", chunk_id="c1")
    c = _doc(page_id="p2", chunk_id="c2")
    assert len(merge_documents([a], [b, c])) == 2


async def test_max_retries_exhausted_always_proceeds(monkeypatch) -> None:
    docs = [_doc(raw=0.10)]  # 근거 전무라도
    out = await _run(monkeypatch, docs, retry=1, max_retries=1)
    assert out["retry_action"] == RetryAction.PROCEED
    assert out["should_re_retrieve"] is False
    assert "gate_flags" in out  # 소진 시 게이트 판정 수행


# ---------------------------------------------------------------------------
# collapse·coverage·채점 단위
# ---------------------------------------------------------------------------


def test_collapse_match_tie_breaks_by_raw_currency() -> None:
    """match 모드 동률(target 부재 → 인접 0.5)은 raw_currency가 승자를 정한다 (v1.3 사다리)."""
    docs = [
        _doc(page_id="d18", doc_key="k8s:doc", site="kubernetes", version="v1.18"),
        _doc(page_id="d29", doc_key="k8s:doc", site="kubernetes", version="v1.29"),
    ]
    from app.agents.trust.node import annotate_version_fits

    lineages = {"k8s:doc": K8S_LINEAGE}
    annotate_version_fits(docs, lineages, ["1.25"], S)  # 둘 다 인접 0.5 동률
    survivors, removed = collapse_siblings(docs, ["1.25"], S)
    assert len(survivors) == 1
    assert survivors[0].product_version == "v1.29"  # raw_currency 0.75 > 0.25
    assert removed["k8s:doc"] == ["v1.18"]


def test_collapse_currency_tie_falls_to_recency() -> None:
    """currency 모드 동률(예: ko/en 중복)은 raw_currency 단계가 무의미 → last_modified로."""
    docs = [
        _doc(page_id="old", doc_key="dd:doc", site="datadog", version="latest", last_modified=_OLD),
        _doc(page_id="new", doc_key="dd:doc", site="datadog", version="latest", last_modified=_RECENT),
    ]
    from app.agents.trust.node import annotate_version_fits

    annotate_version_fits(docs, {"dd:doc": frozenset({"latest"})}, [], S)
    survivors, _ = collapse_siblings(docs, [], S)
    assert survivors[0].page_id == "new"


def test_collapse_empty_doc_key_falls_back_to_page() -> None:
    """계보 없는 문서(라벨 없음)는 page 단위 그룹 — 서로 collapse되지 않는다."""
    docs = [_doc(page_id="p1"), _doc(page_id="p2")]
    survivors, _ = collapse_siblings(docs, [], S)
    assert len(survivors) == 2


def test_coverage_counts_topics_not_documents() -> None:
    docs = [
        _doc(raw=_ABOVE, page_id="a1", doc_key="k8s:a", site="kubernetes", version="v1.33"),
        _doc(raw=_ABOVE, page_id="a2", doc_key="k8s:b", site="kubernetes", version="v1.33"),
    ]
    coverage, waiver, missing, n = compute_coverage(docs, [], S)
    assert n == 2 and coverage == 1.0 and waiver is False and missing == []


def test_coverage_waiver_single_survivor_no_top2() -> None:
    """생존 1건(top2 부재)이면 격차 조건 자동 충족 — top1 조건만 검사 (v1.4)."""
    docs = [_doc(raw=0.95, page_id="solo", doc_key="k8s:solo", site="kubernetes", version="v1.33")]
    coverage, waiver, _, _ = compute_coverage(docs, [], S)
    assert waiver is True and coverage == 1.0


def test_coverage_rerank_fallback_does_not_collapse_to_zero() -> None:
    """리랭커 폴백 시 raw rerank=0이어도 coverage가 0으로 무너지지 않는다 (#202).

    정상 모드: raw 0 → floor 미달 → good 0 → coverage 0 (이게 답변 보류를 유발하던 버그).
    폴백 모드: ratio=0 기본 → 검색 생존 전부 good → distinct 주제 수로 coverage 산정.
    """
    docs = [
        _doc(raw=0.0, page_id="a1", doc_key="k8s:a", site="kubernetes"),
        _doc(raw=0.0, page_id="a2", doc_key="k8s:b", site="kubernetes"),
    ]
    cov_normal, _, _, n_normal = compute_coverage(docs, [], S, rerank_fallback=False)
    assert cov_normal == 0.0 and n_normal == 0  # 기존: rerank 없으면 0 → 보류

    cov_fb, waiver, _, n_fb = compute_coverage(docs, [], S, rerank_fallback=True)
    assert cov_fb == 1.0 and n_fb == 2 and waiver is False  # 폴백: vector 검색 신뢰 → 답변 가능


def test_residual_duplication_counts_unexempt_siblings() -> None:
    """비교 질의가 아니면 형제 혼입이 잔여 중복으로 잡힌다 (collapse 전 호출 가정 검증)."""
    docs = [
        _doc(page_id="a", doc_key="k8s:d", site="kubernetes", version="v1.25"),
        _doc(page_id="b", doc_key="k8s:d", site="kubernetes", version="v1.33"),
    ]
    assert residual_duplication(docs, []) == 0.5  # 1 − 1주제/2(주제,버전)쌍


def test_score_survivors_overall_in_range() -> None:
    docs = [_doc(raw=_ABOVE, page_id="x", doc_key="k8s:x", site="kubernetes", version="v1.33")]
    from app.agents.trust.node import annotate_version_fits

    annotate_version_fits(docs, {"k8s:x": frozenset({"v1.33"})}, [], S)
    out = score_survivors(docs, [], S)
    assert 0.0 <= out.overall <= 1.0
    assert out.owner_trust == 1.0 and out.verification_label == 1.0  # 보고 계약 유지 (중립 상수)


def test_decide_retry_action_pure_proceed_when_sufficient() -> None:
    docs = [
        _doc(raw=_ABOVE, page_id="a", doc_key="k8s:a", site="kubernetes", version="v1.33"),
        _doc(raw=_ABOVE, page_id="b", doc_key="k8s:b", site="kubernetes", version="v1.33"),
    ]
    from app.agents.trust.node import annotate_version_fits

    annotate_version_fits(docs, {}, [], S)
    output = score_survivors(docs, [], S)
    decision = decide_retry_action(docs, [], {}, output, [], S)
    assert decision.action == RetryAction.PROCEED


# ---------------------------------------------------------------------------
# 노드 공통 계약
# ---------------------------------------------------------------------------


async def test_trust_node_emits_trust_score_and_trace(monkeypatch) -> None:
    out = await _run(monkeypatch, [_doc(raw=_ABOVE, doc_key="k8s:a", site="kubernetes", version="v1.33")])
    assert isinstance(out["trust_score"], TrustScore)
    assert out["agent_trace"] == ["trust"]
    assert out["trust_score"].sensitivity_risk == 0.0  # 게이트 전용 관측값


async def test_trust_node_empty_documents_rewrites(monkeypatch) -> None:
    async def _fake_llm(system, user, **kw):
        return "다른 표현의 쿼리"

    monkeypatch.setattr("app.services.llm_selector.call_llm", _fake_llm)
    out = await _run(monkeypatch, [])
    assert out["retry_action"] == RetryAction.REWRITE_QUERY
    assert out["should_re_retrieve"] is True


def test_trust_decision() -> None:
    assert trust_decision({"should_re_retrieve": True}) == "retriever"
    assert trust_decision({"should_re_retrieve": False}) == "answer"
    assert trust_decision({}) == "answer"
