"""Trust Agent 단위 테스트 (규칙기반·결정론, LLM/Qdrant 불필요)."""

from app.agents.state import SourceDocument, TrustScore
from app.agents.trust.node import score_trust, should_re_retrieve, trust_decision, trust_node
from app.config import Settings


def _doc(rerank=0.5, page_id="p1", last_modified="2026-06-07T00:00:00Z", h="h1", content="내용") -> SourceDocument:
    return SourceDocument(
        title="t", content_snippet=content, rerank_score=rerank, page_id=page_id, last_modified=last_modified, hash=h
    )


S = Settings()


def test_score_empty_docs() -> None:
    out = score_trust([], S)
    assert out.overall == 0.0
    assert out.owner_trust == S.trust_owner_neutral
    assert out.verification_label == S.trust_verification_neutral


def test_recency_recent_vs_old() -> None:
    recent = score_trust([_doc(last_modified="2026-06-07T00:00:00Z")], S).recency
    old = score_trust([_doc(last_modified="2018-01-01T00:00:00Z")], S).recency
    assert recent > 0.9
    assert old < 0.1


def test_recency_bad_date_zero() -> None:
    assert score_trust([_doc(last_modified="")], S).recency == 0.0


def test_duplication_same_hash() -> None:
    out = score_trust([_doc(h="x"), _doc(h="x"), _doc(h="y")], S)
    assert out.duplication_conflict == 1 - 2 / 3  # 3개 중 고유 2개


def test_sensitivity_masked_markers() -> None:
    docs = [_doc(content="[MASKED_TOKEN] 값 [MASKED_EMAIL]"), _doc(content="평범")]
    out = score_trust(docs, S)
    assert out.sensitivity_risk == 2 / S.trust_sensitivity_masked_cap


def test_overall_in_range_and_owner_neutral() -> None:
    out = score_trust([_doc()], S)
    assert 0.0 <= out.overall <= 1.0
    assert out.owner_trust == 1.0 and out.verification_label == 1.0


def test_gate_conflicting() -> None:
    # 서로 다른 page, top 점수 차 < gap(0.05) → 충돌
    docs = [_doc(page_id="p1", rerank=0.9), _doc(page_id="p2", rerank=0.88)]
    assert score_trust(docs, S).gate_conflicting is True
    # 점수 차 큼 → 충돌 아님
    docs2 = [_doc(page_id="p1", rerank=0.9), _doc(page_id="p2", rerank=0.3)]
    assert score_trust(docs2, S).gate_conflicting is False


def test_should_re_retrieve() -> None:
    assert should_re_retrieve([], S, retry_count=0, max_retries=1) is True  # 문서 0
    assert should_re_retrieve([_doc(rerank=0.1)], S, 0, 1) is True  # top < τ(0.288)
    assert should_re_retrieve([_doc(rerank=0.9)], S, 0, 1) is False  # 충분
    assert should_re_retrieve([_doc(rerank=0.1)], S, retry_count=1, max_retries=1) is False  # 한도 초과


async def test_trust_node_triggers_retry() -> None:
    state = {"documents": [_doc(rerank=0.1)], "retry_count": 0, "max_retries": 1}
    out = await trust_node(state)
    assert isinstance(out["trust_score"], TrustScore)
    assert out["should_re_retrieve"] is True
    assert out["retry_count"] == 1
    assert out["domain"] is None  # 재시도 시 도메인 해제
    assert out["agent_trace"] == ["trust"]


async def test_trust_node_no_retry_passes_to_answer() -> None:
    state = {"documents": [_doc(rerank=0.9)], "retry_count": 0, "max_retries": 1}
    out = await trust_node(state)
    assert out["should_re_retrieve"] is False
    assert "domain" not in out  # 도메인 유지
    assert out["gate_flags"] is not None


def test_trust_decision() -> None:
    assert trust_decision({"should_re_retrieve": True}) == "retriever"
    assert trust_decision({"should_re_retrieve": False}) == "answer"
    assert trust_decision({}) == "answer"
