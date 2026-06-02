"""Answer Agent 단위 테스트 (LLM mock 사용)."""

import json

import pytest

from app.agents.answer import node as node_mod
from app.agents.answer.answerability import GateFlags, decide_answerability
from app.agents.answer.node import answer_node
from app.agents.state import AnswerabilityStatus, FiveElements, SourceDocument


def _doc(title: str = "제목", content: str = "내용") -> SourceDocument:
    return SourceDocument(title=title, content_snippet=content)


def _mock_llm(response: str):
    async def _call(*args, **kwargs):
        return response

    return _call


def _ans_json(status: str = "answerable", indices: tuple[int, ...] = (0,)) -> str:
    return json.dumps(
        {
            "situation": "상황",
            "cause": "원인",
            "evidence": "근거",
            "solution": "해결",
            "infra_context": "맥락",
            "answerability_status": status,
            "answerability_reason": "",
            "source_indices": list(indices),
        }
    )


@pytest.mark.asyncio
async def test_answer_with_documents(monkeypatch):
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_ans_json("answerable", (0,))))
    out = await answer_node({"refined_query": "q", "documents": [_doc()]})
    assert out["answerability_status"] == AnswerabilityStatus.ANSWERABLE
    assert out["answer"].situation != ""
    assert len(out["sources"]) == 1
    assert out["agent_trace"] == ["answer"]


@pytest.mark.asyncio
async def test_answer_no_documents_skips_llm(monkeypatch):
    async def _fail(*args, **kwargs):
        raise AssertionError("문서 0건이면 LLM을 호출하면 안 된다")

    monkeypatch.setattr(node_mod, "call_llm", _fail)
    out = await answer_node({"refined_query": "q", "documents": []})
    assert out["answerability_status"] == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE
    assert out["answerability_reason"]
    assert out["answer"] == FiveElements()


@pytest.mark.asyncio
async def test_answer_not_enough_by_llm_holds(monkeypatch):
    # LLM이 not_enough → 5요소 비우고 보류
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_ans_json("not_enough_evidence", ())))
    out = await answer_node({"refined_query": "q", "documents": [_doc()]})
    assert out["answerability_status"] == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE
    assert out["answer"] == FiveElements()
    assert out["sources"] == []


@pytest.mark.asyncio
async def test_answer_partially_keeps_five(monkeypatch):
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_ans_json("partially_answerable", (0,))))
    out = await answer_node({"refined_query": "q", "documents": [_doc()]})
    assert out["answerability_status"] == AnswerabilityStatus.PARTIALLY_ANSWERABLE
    assert out["answer"].situation != ""  # 부분 답변 유지
    assert out["answerability_reason"]


@pytest.mark.asyncio
async def test_answer_citation_guard_demotes(monkeypatch):
    # LLM이 answerable이지만 인용 source 0건 → PARTIALLY로 강등
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_ans_json("answerable", ())))
    out = await answer_node({"refined_query": "q", "documents": [_doc()]})
    assert out["answerability_status"] == AnswerabilityStatus.PARTIALLY_ANSWERABLE
    assert out["sources"] == []
    assert out["answer"].situation != ""  # 강등이라 답변은 유지


@pytest.mark.asyncio
async def test_answer_parse_error_holds(monkeypatch):
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm("이건 JSON이 아님"))
    out = await answer_node({"refined_query": "q", "documents": [_doc()]})
    assert out["answerability_status"] == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE
    assert "파싱" in out["answerability_reason"] or "생성 실패" in out["answerability_reason"]


@pytest.mark.asyncio
async def test_answer_llm_failure_holds(monkeypatch):
    async def _boom(*args, **kwargs):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(node_mod, "call_llm", _boom)
    out = await answer_node({"refined_query": "q", "documents": [_doc()]})
    assert out["answerability_status"] == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE
    assert "LLM down" in out["error"]


@pytest.mark.asyncio
async def test_answer_coerces_list_solution(monkeypatch):
    # LLM이 solution을 배열(단계 목록)로 줘도 문자열로 수용해야 한다 (실 LLM 흔한 패턴)
    payload = {
        "situation": "상황",
        "cause": "원인",
        "evidence": "근거",
        "solution": ["1. 첫 단계", "2. 둘째 단계"],
        "infra_context": "맥락",
        "answerability_status": "answerable",
        "source_indices": [0],
    }
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(json.dumps(payload)))
    out = await answer_node({"refined_query": "q", "documents": [_doc()]})
    assert out["answerability_status"] == AnswerabilityStatus.ANSWERABLE
    assert "1. 첫 단계" in out["answer"].solution
    assert "2. 둘째 단계" in out["answer"].solution


@pytest.mark.asyncio
async def test_answer_source_mapping(monkeypatch):
    docs = [_doc("A"), _doc("B"), _doc("C")]
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_ans_json("answerable", (0, 2))))
    out = await answer_node({"refined_query": "q", "documents": docs})
    assert len(out["sources"]) == 2
    assert out["sources"][0].title == "A"
    assert out["sources"][1].title == "C"


@pytest.mark.asyncio
async def test_answer_unknown_status_maps_not_enough(monkeypatch):
    # LLM이 enum에 없는 status를 줘도 ValidationError로 5요소를 통째로 버리지 않고 NOT_ENOUGH로 안전 매핑
    payload = {
        "situation": "상황",
        "cause": "원인",
        "evidence": "근거",
        "solution": "해결",
        "infra_context": "맥락",
        "answerability_status": "totally_unknown_state",
        "source_indices": [0],
    }
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(json.dumps(payload)))
    out = await answer_node({"refined_query": "q", "documents": [_doc()]})
    assert out["answerability_status"] == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE


# ── decide_answerability 판단 경계 단위 (P1 점수·게이트 경로 미리 검증) ──
def test_decide_gate_conflicting_outdated():
    docs = [_doc()]
    assert decide_answerability(docs, gate=GateFlags(conflicting=True)) == AnswerabilityStatus.CONFLICTING_EVIDENCE
    assert decide_answerability(docs, gate=GateFlags(deprecated_only=True)) == AnswerabilityStatus.OUTDATED_EVIDENCE


def test_decide_p1_score_thresholds():
    docs = [_doc()]
    assert decide_answerability(docs, evidence_score=0.85) == AnswerabilityStatus.ANSWERABLE
    assert decide_answerability(docs, evidence_score=0.60) == AnswerabilityStatus.PARTIALLY_ANSWERABLE
    assert decide_answerability(docs, evidence_score=0.40) == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE


def test_decide_floor_no_docs_overrides_llm():
    # 문서 0건이면 LLM이 answerable이라 해도 보류
    assert (
        decide_answerability([], llm_status=AnswerabilityStatus.ANSWERABLE) == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE
    )
