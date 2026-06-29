import pytest

from app.agents.retriever import node as node_mod
from app.agents.retriever import search as search_mod
from app.agents.retriever.node import retrieve_node
from app.agents.state import SourceDocument
from app.config import Settings


def _hit(chunk_id, content, score, domain="장애대응", site="", product_version="", doc_key="", is_eol=False):
    payload = {
        "chunk_id": chunk_id,
        "content": content,
        "page_title": "제목",
        "source_url": "http://x",
        "space_key": "OnRamp",
        "domain": domain,
        "last_modified": "",
        "site": site,
        "product_version": product_version,
        "doc_key": doc_key,
        "is_eol": is_eol,
    }
    return type("SP", (), {"id": chunk_id, "payload": payload, "score": score})()


class _FakeEmbedder:
    async def embed_query(self, text):
        return [0.1, 0.2, 0.3]


def _patch(monkeypatch, search_fn, rerank_obj, lineages=None):
    monkeypatch.setattr(node_mod, "get_embedder", lambda *a, **k: _FakeEmbedder())
    # node는 search_with_mode를 거쳐 dense_search를 호출 → search 모듈의 dense_search를 패치
    monkeypatch.setattr(search_mod, "dense_search", search_fn)
    monkeypatch.setattr(node_mod, "get_reranker", lambda *a, **k: rerank_obj)
    # 계보 조회는 Qdrant facet → 스텁으로 차단 (기본: 전부 계보 없음 = version_fit 중립)
    stub = lineages or {}
    monkeypatch.setattr(node_mod, "get_lineages", lambda keys, **kw: {k: stub.get(k, frozenset()) for k in keys})


@pytest.mark.asyncio
async def test_node_maps_to_source_document(monkeypatch):
    hits = [_hit("c1", "alpha", 0.9), _hit("c2", "beta", 0.8)]

    async def fake_search(qv, top_k, *, domain=None, **k):
        return hits

    class _R:
        def rerank(self, q, cands):
            return [(0.5, p) for _, p in cands]

    _patch(monkeypatch, fake_search, _R())
    out = await retrieve_node({"refined_query": "q", "domains": ["장애대응"]})
    docs = out["documents"]
    assert out["agent_trace"] == ["retriever"]
    assert all(isinstance(d, SourceDocument) for d in docs)
    assert docs[0].title == "제목"
    assert docs[0].content_snippet == "alpha"
    assert docs[0].score == 0.9


@pytest.mark.asyncio
async def test_node_keeps_late_exact_fact_in_chunk_context(monkeypatch):
    content = f"{'prefix ' * 100}Schema MUST be between -4 and 8 inclusive."

    async def fake_search(qv, top_k, *, domain=None, **k):
        return [_hit("c1", content, 0.9)]

    class _R:
        def rerank(self, q, cands):
            return [(0.99, p) for _, p in cands]

    _patch(monkeypatch, fake_search, _R())
    out = await retrieve_node({"refined_query": "Schema 범위", "domains": []})
    assert "-4 and 8" in out["documents"][0].content_snippet


@pytest.mark.asyncio
async def test_node_domain_filter_fallback(monkeypatch):
    calls = []

    async def fake_search(qv, top_k, *, domain=None, **k):
        calls.append(domain)
        return [] if domain else [_hit("c1", "x", 0.7)]

    class _R:
        def rerank(self, q, cands):
            return [(0.1, p) for _, p in cands]

    _patch(monkeypatch, fake_search, _R())
    monkeypatch.setattr(node_mod, "get_settings", lambda: Settings(retriever_domain_filter_mode="hybrid"))
    out = await retrieve_node({"refined_query": "q", "domains": ["장애대응"]})
    assert calls == ["장애대응", None]  # hybrid: 필터→0건→무필터 재검색
    assert len(out["documents"]) == 1


@pytest.mark.asyncio
async def test_node_soft_default_no_filter_but_bonus(monkeypatch):
    """기본 soft: 도메인으로 필터하지 않고(무필터 검색) 일치 시 가산만 적용."""
    calls = []

    async def fake_search(qv, top_k, *, domain=None, **k):
        calls.append(domain)
        return [_hit("c1", "a", 0.9, domain="api_reference"), _hit("c2", "b", 0.9, domain="manual")]

    class _R:
        def rerank(self, q, cands):
            return [(0.5, p) for _, p in cands]

    _patch(monkeypatch, fake_search, _R())  # 기본값 soft
    out = await retrieve_node({"refined_query": "q", "domains": ["manual"]})
    assert calls == [None]  # soft → 무필터 검색
    assert out["documents"][0].content_snippet == "b"  # manual 일치 가산으로 우선


@pytest.mark.asyncio
async def test_node_fallback_applies_domain_bonus(monkeypatch):
    """리랭커 실패 폴백에서도 soft 도메인 가산이 정렬에 반영된다."""

    async def fake_search(qv, top_k, *, domain=None, **k):
        # 불일치 문서가 vec 약간 높지만, 일치 문서가 가산으로 역전
        return [_hit("c1", "other", 0.55, domain="api_reference"), _hit("c2", "match", 0.5, domain="manual")]

    class _R:
        def rerank(self, q, cands):
            raise RuntimeError("OOM")

    _patch(monkeypatch, fake_search, _R())  # 기본값 soft
    out = await retrieve_node({"refined_query": "q", "domains": ["manual"]})
    # 폴백: 0.55(api_reference) vs 0.5+0.1(manual 가산)=0.6 → manual 먼저
    assert out["documents"][0].content_snippet == "match"
    assert out["documents"][0].rerank_score == 0.0


@pytest.mark.asyncio
async def test_node_rerank_oom_fallback(monkeypatch):
    hits = [_hit("c1", "a", 0.3), _hit("c2", "b", 0.9)]

    async def fake_search(qv, top_k, *, domain=None, **k):
        return hits

    class _R:
        def rerank(self, q, cands):
            raise RuntimeError("OOM")

    _patch(monkeypatch, fake_search, _R())
    out = await retrieve_node({"refined_query": "q", "domains": ["장애대응"]})
    # 리랭커 실패 → vector score 순 폴백 (c2=0.9 먼저)
    assert out["documents"][0].content_snippet == "b"
    assert out["documents"][0].rerank_score == 0.0


@pytest.mark.asyncio
async def test_node_rerank_missing_dependency_fallback(monkeypatch):
    """sentence-transformers 미설치(ModuleNotFoundError)도 vector score 순 폴백."""
    hits = [_hit("c1", "a", 0.3), _hit("c2", "b", 0.9)]

    async def fake_search(qv, top_k, *, domain=None, **k):
        return hits

    class _R:
        def rerank(self, q, cands):
            raise ModuleNotFoundError("No module named 'sentence_transformers'")

    _patch(monkeypatch, fake_search, _R())
    out = await retrieve_node({"refined_query": "q", "domains": ["장애대응"]})
    assert out["documents"][0].content_snippet == "b"  # vec score 0.9 우선
    assert out["documents"][0].rerank_score == 0.0


@pytest.mark.asyncio
async def test_node_low_quality_filtered_expands_and_recovers(monkeypatch):
    """filtered 결과가 비어있지 않아도 저품질(최고 score < 임계)이면 무필터 확장 → 정답 회수."""
    calls = []

    async def fake_search(qv, top_k, *, domain=None, **k):
        calls.append(domain)
        # filtered: 저품질 오답(0.3만) / unfiltered: 오답 + 정답(0.9)
        return [_hit("c1", "wrong", 0.3)] if domain else [_hit("c1", "wrong", 0.3), _hit("c2", "right", 0.9)]

    class _R:
        def rerank(self, q, cands):
            return [(1.0 if "right" in text else 0.1, p) for text, p in cands]

    _patch(monkeypatch, fake_search, _R())
    monkeypatch.setattr(node_mod, "get_settings", lambda: Settings(retriever_domain_filter_mode="hybrid"))
    out = await retrieve_node({"refined_query": "q", "domains": ["manual"]})
    assert calls == ["manual", None]  # hybrid: 저품질 filtered → 무필터 확장
    assert out["documents"][0].content_snippet == "right"  # merge 후 정답 회수


@pytest.mark.asyncio
async def test_node_high_quality_filtered_no_expand(monkeypatch):
    """filtered 최고 score가 임계 이상이면 무필터 확장하지 않는다."""
    calls = []

    async def fake_search(qv, top_k, *, domain=None, **k):
        calls.append(domain)
        return [_hit("c1", "a", 0.9)]

    class _R:
        def rerank(self, q, cands):
            return [(0.5, p) for _, p in cands]

    _patch(monkeypatch, fake_search, _R())
    monkeypatch.setattr(node_mod, "get_settings", lambda: Settings(retriever_domain_filter_mode="hybrid"))
    await retrieve_node({"refined_query": "q", "domains": ["manual"]})
    assert calls == ["manual"]  # hybrid 고품질 → 확장 없음


@pytest.mark.asyncio
async def test_node_domain_match_bonus(monkeypatch):
    """기저 rerank 점수가 같으면 도메인 일치 문서가 가산으로 우선된다."""

    async def fake_search(qv, top_k, *, domain=None, **k):
        return [_hit("c1", "a", 0.9, domain="api_reference"), _hit("c2", "b", 0.9, domain="manual")]

    class _R:
        def rerank(self, q, cands):
            return [(0.5, p) for _, p in cands]  # 동일 기저 점수

    _patch(monkeypatch, fake_search, _R())
    out = await retrieve_node({"refined_query": "q", "domains": ["manual"]})
    assert out["documents"][0].content_snippet == "b"  # domain=manual 일치 → 가산으로 먼저


@pytest.mark.asyncio
async def test_node_explicit_empty_domains_no_single_fallback(monkeypatch):
    """명시적 domains=[](예: Trust 재검색 초기화)는 구형 단수 domain으로 복구하지 않는다 → 가산 없음."""

    async def fake_search(qv, top_k, *, domain=None, **k):
        return [_hit("c1", "a", 0.9, domain="api_reference"), _hit("c2", "b", 0.9, domain="manual")]

    class _R:
        def rerank(self, q, cands):
            return [(0.5, p) for _, p in cands]  # 동일 기저 점수

    _patch(monkeypatch, fake_search, _R())
    # domains=[] 명시 + 구형 domain="manual" 동시 존재 → domains=[]를 존중(가산 0), 입력 순서 유지
    out = await retrieve_node({"refined_query": "q", "domains": [], "domain": "manual"})
    assert out["documents"][0].content_snippet == "a"  # 가산 없음 → 입력 순서 그대로(c1=a 먼저)


@pytest.mark.asyncio
async def test_node_legacy_single_domain_fallback(monkeypatch):
    """domains 키가 아예 없으면 구형 단수 domain으로 폴백해 가산이 동작한다."""

    async def fake_search(qv, top_k, *, domain=None, **k):
        return [_hit("c1", "a", 0.9, domain="api_reference"), _hit("c2", "b", 0.9, domain="manual")]

    class _R:
        def rerank(self, q, cands):
            return [(0.5, p) for _, p in cands]

    _patch(monkeypatch, fake_search, _R())
    out = await retrieve_node({"refined_query": "q", "domain": "manual"})  # domains 키 없음
    assert out["documents"][0].content_snippet == "b"  # 단수 폴백 → manual 가산


# ── 점수 분리 + 버전·권위 부스트 (#103) ─────────────────────────────


@pytest.mark.asyncio
async def test_node_separates_raw_and_ranking_scores(monkeypatch):
    """raw(τ 진단)와 ranking(부스트 정렬)이 분리 저장된다 — raw는 [0,1], ranking ≥ raw."""

    async def fake_search(qv, top_k, *, domain=None, **k):
        return [_hit("c1", "a", 0.9, domain="manual", site="apache")]

    class _R:
        def rerank(self, q, cands):
            return [(0.8, p) for _, p in cands]

    _patch(monkeypatch, fake_search, _R())
    out = await retrieve_node({"refined_query": "q", "domains": ["manual"]})
    doc = out["documents"][0]
    assert doc.raw_rerank_score == 0.8  # cross-encoder 원점수 그대로
    assert doc.rerank_score > doc.raw_rerank_score  # 부스트(domain+version+authority) 합산


@pytest.mark.asyncio
async def test_node_version_boost_reorders_eol_sibling(monkeypatch):
    """raw 동률인 버전 형제 중 EOL(2.2)이 최신(2.4)보다 뒤로 밀린다 — 설계 7.4."""
    lineages = {"apache:cn": frozenset({"2.2", "2.4"})}

    async def fake_search(qv, top_k, *, domain=None, **k):
        return [
            _hit("c-old", "old", 0.9, site="apache", product_version="2.2", doc_key="apache:cn", is_eol=True),
            _hit("c-new", "new", 0.9, site="apache", product_version="2.4", doc_key="apache:cn"),
        ]

    class _R:
        def rerank(self, q, cands):
            return [(0.7, p) for _, p in cands]  # raw 동률

    _patch(monkeypatch, fake_search, _R(), lineages=lineages)
    out = await retrieve_node({"refined_query": "q", "domains": []})
    assert out["documents"][0].content_snippet == "new"  # version_fit 1.0 vs EOL 캡 0.3
    assert out["documents"][0].raw_rerank_score == out["documents"][1].raw_rerank_score == 0.7


@pytest.mark.asyncio
async def test_node_fallback_zeroes_both_scores(monkeypatch):
    """리랭커 폴백은 raw/ranking 둘 다 0.0 (리랭킹 미수행 신호 — τ 진단도 폴백 인지)."""

    async def fake_search(qv, top_k, *, domain=None, **k):
        return [_hit("c1", "a", 0.9)]

    class _R:
        def rerank(self, q, cands):
            raise RuntimeError("OOM")

    _patch(monkeypatch, fake_search, _R())
    out = await retrieve_node({"refined_query": "q", "domains": []})
    doc = out["documents"][0]
    assert doc.rerank_score == 0.0
    assert doc.raw_rerank_score == 0.0


@pytest.mark.asyncio
async def test_node_maps_lineage_meta_fields(monkeypatch):
    """payload의 버전 계보 메타(#94)가 SourceDocument로 매핑된다."""

    async def fake_search(qv, top_k, *, domain=None, **k):
        return [_hit("c1", "a", 0.9, site="apache", product_version="2.2", doc_key="apache:cn", is_eol=True)]

    class _R:
        def rerank(self, q, cands):
            return [(0.5, p) for _, p in cands]

    _patch(monkeypatch, fake_search, _R())
    out = await retrieve_node({"refined_query": "q", "domains": []})
    doc = out["documents"][0]
    assert doc.chunk_id == "c1"
    assert doc.site == "apache"
    assert doc.product_version == "2.2"
    assert doc.doc_key == "apache:cn"
    assert doc.is_eol is True


@pytest.mark.asyncio
async def test_node_honors_config_filter_mode_hard(monkeypatch):
    """config가 hard면 filtered가 저품질이어도 무필터 확장하지 않는다."""
    calls = []

    async def fake_search(qv, top_k, *, domain=None, **k):
        calls.append(domain)
        return [_hit("c1", "x", 0.2)]  # 저품질

    class _R:
        def rerank(self, q, cands):
            return [(0.5, p) for _, p in cands]

    _patch(monkeypatch, fake_search, _R())
    monkeypatch.setattr(node_mod, "get_settings", lambda: Settings(retriever_domain_filter_mode="hard"))
    await retrieve_node({"refined_query": "q", "domains": ["manual"]})
    assert calls == ["manual"]  # hard → 확장 없음


# ── 재검색 사다리 소비 (#108) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_node_expand_topics_doubles_top_k_and_excludes(monkeypatch):
    from app.agents.state import RetryAction

    seen = {}

    async def fake_search(qv, top_k, *, domain=None, filters=None, **k):
        seen["top_k"] = top_k
        seen["filters"] = filters
        return [_hit("c1", "a", 0.9)]

    class _R:
        def rerank(self, q, cands):
            return [(0.9, p) for _, p in cands]

    monkeypatch.setattr(node_mod, "get_embedder", lambda *a, **k: _FakeEmbedder())
    monkeypatch.setattr(node_mod, "search_with_mode", fake_search)
    monkeypatch.setattr(node_mod, "get_reranker", lambda *a, **k: _R())
    monkeypatch.setattr(node_mod, "get_lineages", lambda keys, **kw: {k: frozenset() for k in keys})

    out = await retrieve_node(
        {
            "refined_query": "q",
            "domains": [],
            "retry_action": RetryAction.EXPAND_TOPICS,
            "excluded_doc_keys": ["apache:done"],
        }
    )
    s = Settings()
    assert seen["top_k"] == s.retriever_top_k * 2  # 주제 확장: 후보 풀 2배
    assert seen["filters"].excluded_doc_keys == ("apache:done",)
    assert out["documents"]


@pytest.mark.asyncio
async def test_node_retry_version_passes_filters(monkeypatch):
    from app.agents.state import RetryAction

    seen = {}

    async def fake_search(qv, top_k, *, domain=None, filters=None, **k):
        seen["filters"] = filters
        return [_hit("c1", "a", 0.9)]

    class _R:
        def rerank(self, q, cands):
            return [(0.9, p) for _, p in cands]

    monkeypatch.setattr(node_mod, "get_embedder", lambda *a, **k: _FakeEmbedder())
    monkeypatch.setattr(node_mod, "search_with_mode", fake_search)
    monkeypatch.setattr(node_mod, "get_reranker", lambda *a, **k: _R())
    monkeypatch.setattr(node_mod, "get_lineages", lambda keys, **kw: {k: frozenset() for k in keys})

    await retrieve_node(
        {
            "refined_query": "q",
            "domains": [],
            "retry_action": RetryAction.RETRY_VERSION,
            "version_filter": "2.4",
            "pinned_doc_keys": ["apache:mpm"],
        }
    )
    assert seen["filters"].version == "2.4"
    assert seen["filters"].pinned_doc_keys == ("apache:mpm",)


# ── #135 rerank 커스텀 span ──
def _fake_span(captured):
    from contextlib import contextmanager

    class _Span:
        def update(self, **kw):
            captured.update(kw)

    @contextmanager
    def _cm(**kw):
        captured["start"] = kw
        yield _Span()

    return _cm


@pytest.mark.asyncio
async def test_rerank_span_records_metadata_on_success(monkeypatch):
    hits = [_hit("c1", "a", 0.9)]

    async def fake_search(qv, top_k, *, domain=None, **k):
        return hits

    class _R:
        def rerank(self, q, c):
            return [(0.7, p) for _, p in c]

    _patch(monkeypatch, fake_search, _R())
    captured: dict = {}
    monkeypatch.setattr(node_mod, "langfuse_span", _fake_span(captured))

    await retrieve_node({"refined_query": "q", "domains": ["장애대응"]})

    md = captured["metadata"]
    assert md["backend"] in ("torch", "onnx", "remote")
    assert md["reranked"] is True and md["fallback"] is None
    assert md["n_hits"] == 1 and md["zero_hit"] is False
    assert md["top_raw_score"] == 0.7


@pytest.mark.asyncio
async def test_rerank_span_records_fallback_on_error(monkeypatch):
    hits = [_hit("c1", "a", 0.9)]

    async def fake_search(qv, top_k, *, domain=None, **k):
        return hits

    class _R:
        def rerank(self, q, c):
            raise RuntimeError("gpu down")

    _patch(monkeypatch, fake_search, _R())
    captured: dict = {}
    monkeypatch.setattr(node_mod, "langfuse_span", _fake_span(captured))

    out = await retrieve_node({"refined_query": "q", "domains": ["장애대응"]})

    assert out["documents"]  # 폴백으로 결과는 나옴
    assert captured["metadata"]["fallback"] == "error"
    assert captured["metadata"]["reranked"] is False
