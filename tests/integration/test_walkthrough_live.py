"""Trust 재설계(#108) 워크스루 라이브 E2E (opt-in — 실 Qdrant + OpenAI 비용 발생).

설계 문서(docs/Baemin/01_trust_agent_redesign.md) 11장 워크스루를 **실 코퍼스**
(재색인 908페이지 + 라벨 파생 버전 메타)에서 그래프 전체로 검증한다.
단위 회귀(tests/unit/test_trust_node.py)와 달리 Router LLM 추출·리랭커 점수 분포·
계보 facet까지 실물로 통과한다.

LLM은 비결정적이므로 **하드 불변식만 단언**한다 (점수 절대값·문구 단언 금지).
실행: pytest tests/integration/test_walkthrough_live.py -v
"""

import os

import pytest

from app.agents.graph import compiled_graph
from app.agents.state import AnswerabilityStatus, RetryAction
from app.config import get_settings
from app.db.qdrant import get_qdrant
from app.rag.labels import versions_equal

# 키는 env 또는 .env(pydantic Settings) 어느 쪽으로 와도 인정
pytestmark = pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") or get_settings().openai_api_key),
    reason="실 LLM 키(OPENAI_API_KEY) 필요",
)


@pytest.fixture(autouse=True)
def _require_qdrant():
    try:
        get_qdrant().get_collections()
    except Exception:
        pytest.skip("Qdrant 미가동 (make up 필요)")


async def test_walkthrough_a_single_fact_version_query() -> None:
    """A. 버전 명시 단일 사실 질의 — collapse + match 모드 + 재검색 없음."""
    r = await compiled_graph.ainvoke({"query": "Kubernetes v1.33에서 초기화 컨테이너 디버그하는 방법 알려줘"})

    assert any(versions_equal(v, "1.33") for v in r.get("target_versions", []))  # Router 추출
    assert r["answerability_status"] == AnswerabilityStatus.ANSWERABLE
    assert r["retry_action"] == RetryAction.PROCEED
    assert r["agent_trace"].count("retriever") == 1  # waiver/충분 근거 → 재검색 없음
    # collapse: 같은 doc_key(버전 형제)는 1건만 — 컨텍스트에 같은 주제 중복 없음
    keys = [d.doc_key for d in r["documents"] if d.doc_key]
    assert len(keys) == len(set(keys))
    # 생존 인용은 target 버전 문서가 우선 (match fit 1.0)
    top = r["documents"][0]
    assert versions_equal(top.product_version, "1.33")
    assert r["gate_flags"].conflicting is False  # 상호보완 문서 충돌 오탐 없음 (gap 보정 검증)


async def test_walkthrough_b_comparison_keeps_both_versions() -> None:
    """B. 버전 비교 질의 — collapse 면제로 양 버전 잔류 + 회수율 coverage."""
    r = await compiled_graph.ainvoke(
        {"query": "Kubernetes 1.25와 1.33 각각에서 초기화 컨테이너 디버그하는 방법 알려줘"}
    )

    targets = r.get("target_versions", [])
    assert len(targets) == 2  # 비교 질의 추출
    versions = {d.product_version for d in r["documents"]}
    assert any(versions_equal(v, "1.25") for v in versions)  # 면제 — 두 버전 모두 컨텍스트에
    assert any(versions_equal(v, "1.33") for v in versions)
    assert r["trust_score"].coverage == 1.0  # 회수율 2/2
    assert r["missing_versions"] == []
    assert r["answerability_status"] == AnswerabilityStatus.ANSWERABLE


async def test_walkthrough_d_no_conflict_false_positive() -> None:
    """D. 일반 질의에서 충돌 게이트 오탐 금지 — 상호보완 문서·미결합 형제 모두."""
    r = await compiled_graph.ainvoke({"query": "Apache 프록시 설정 방법 알려줘"})

    assert r["gate_flags"].conflicting is False
    assert r["answerability_status"] != AnswerabilityStatus.CONFLICTING_EVIDENCE
    assert r["documents"]  # 답변 근거는 존재 (상태는 PARTIALLY/ANSWERABLE 비결정 허용)


async def test_walkthrough_no_evidence_rewrites_then_holds_honestly() -> None:
    """무근거 질의 — 사다리 1행(쿼리 재작성) 발동 후에도 부족하면 정직한 보류.

    코퍼스에 '버전 간 차이'를 다루는 문서가 없다 — 버전별 how-to만 존재.
    재작성 쿼리는 비결정적이므로 '재검색이 1회 일어났다'는 불변식만 단언한다.
    """
    r = await compiled_graph.ainvoke({"query": "Kubernetes 1.25에서 1.33으로 올리면 디버깅 방법이 뭐가 달라져?"})

    assert r["agent_trace"].count("retriever") == 2  # 사다리 재검색 정확히 1회 (max_retries=1)
    assert r["retry_count"] == 1
    # 재작성이 우연히 근거를 찾으면 PARTIALLY까지 허용 — 단 근거 없이 ANSWERABLE은 금지
    if r["answerability_status"] == AnswerabilityStatus.ANSWERABLE:
        assert r["documents"], "근거 없이 ANSWERABLE 금지"
