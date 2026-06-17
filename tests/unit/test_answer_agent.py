"""Answer Agent 단위 테스트 (LLM mock 사용)."""

import json

import pytest

from app.agents.answer import node as node_mod
from app.agents.answer.answerability import GateFlags, decide_answerability
from app.agents.answer.formatter import format_answer, format_freeform
from app.agents.answer.node import answer_node
from app.agents.format_policy import decide_answer_format
from app.agents.state import AnswerabilityStatus, Domain, FiveElements, SourceDocument

# 포맷은 라우터 domains 기준(#191). 구조화 경로 테스트는 incident를 명시한다.
_INCIDENT = [Domain.INCIDENT]


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


def _free_json(status: str = "answerable", indices: tuple[int, ...] = (0,), text: str = "자유 형식 답변입니다.") -> str:
    return json.dumps(
        {
            "answer_text": text,
            "answerability_status": status,
            "answerability_reason": "",
            "source_indices": list(indices),
        }
    )


# ── 포맷 결정(#191) — 라우터 domains 단독 ──────────────────────────────


def test_decide_answer_format_router_only():
    s = {"incident"}
    assert decide_answer_format([Domain.INCIDENT], s) == "structured"
    assert decide_answer_format([Domain.MANUAL], s) == "freeform"
    assert decide_answer_format([Domain.PLANNING, Domain.MEETING_NOTE], s) == "freeform"
    assert decide_answer_format([Domain.INCIDENT, Domain.MANUAL], s) == "structured"  # 교집합 있으면 structured
    assert decide_answer_format([], s) == "freeform"  # 라우터 애매 → freeform


@pytest.mark.asyncio
async def test_answer_format_uses_router_value_over_mutated_domains(monkeypatch):
    # 라우터가 박은 answer_format이 우선 — Trust가 domains를 비워도(EXPAND_TOPICS) 포맷 불변 (#191 E2E 버그)
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_ans_json("answerable", (0,))))
    out = await answer_node({"refined_query": "q", "documents": [_doc()], "domains": [], "answer_format": "structured"})
    assert out["answer_format"] == "structured"  # domains=[] 인데도 라우터 값(structured) 유지
    assert out["answer"].situation != ""


def test_structured_answer_domains_normalized():
    # env 오설정(대소문자·공백)이 들어와도 Domain 키와 교집합이 깨지지 않도록 정규화 (CodeRabbit #192)
    from app.config import Settings

    s = Settings(structured_answer_domains={" Incident ", "MANUAL", ""})
    assert s.structured_answer_domains == {"incident", "manual"}


@pytest.mark.asyncio
async def test_incident_uses_structured(monkeypatch):
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_ans_json("answerable", (0,))))
    out = await answer_node({"refined_query": "q", "documents": [_doc()], "domains": _INCIDENT})
    assert out["answer_format"] == "structured"
    assert out["answer"].situation != ""
    assert out["answer_text"] == ""


@pytest.mark.asyncio
async def test_non_incident_uses_freeform(monkeypatch):
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_free_json("answerable", (0,))))
    out = await answer_node({"refined_query": "q", "documents": [_doc()], "domains": [Domain.MANUAL]})
    assert out["answer_format"] == "freeform"
    assert out["answer_text"] != ""
    assert out["answer"] == FiveElements()  # 5요소는 비움
    assert len(out["sources"]) == 1
    assert out["answerability_status"] == AnswerabilityStatus.ANSWERABLE


@pytest.mark.asyncio
async def test_freeform_holds_on_not_enough(monkeypatch):
    # freeform도 grounding·answerability 동일 — not_enough면 본문 비우고 보류
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_free_json("not_enough_evidence", ())))
    out = await answer_node({"refined_query": "q", "documents": [_doc()], "domains": [Domain.MEETING_NOTE]})
    assert out["answer_format"] == "freeform"
    assert out["answer_text"] == ""
    assert out["sources"] == []
    assert out["answerability_status"] == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE


@pytest.mark.asyncio
async def test_freeform_citation_guard_demotes(monkeypatch):
    # freeform도 answerable인데 인용 0건 → PARTIALLY 강등 (구조화와 동일 guard)
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_free_json("answerable", ())))
    out = await answer_node({"refined_query": "q", "documents": [_doc()], "domains": [Domain.API_REFERENCE]})
    assert out["answerability_status"] == AnswerabilityStatus.PARTIALLY_ANSWERABLE
    assert out["answer_text"] != ""  # 강등이라 답변 유지
    assert out["sources"] == []


# ── 구조화(incident) 경로 — 기존 동작 보존 ──────────────────────────────


@pytest.mark.asyncio
async def test_answer_with_documents(monkeypatch):
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_ans_json("answerable", (0,))))
    out = await answer_node({"refined_query": "q", "documents": [_doc()], "domains": _INCIDENT})
    assert out["answerability_status"] == AnswerabilityStatus.ANSWERABLE
    assert out["answer"].situation != ""
    assert len(out["sources"]) == 1
    assert out["agent_trace"] == ["answer"]


@pytest.mark.asyncio
async def test_answer_no_documents_skips_llm(monkeypatch):
    async def _fail(*args, **kwargs):
        raise AssertionError("문서 0건이면 LLM을 호출하면 안 된다")

    monkeypatch.setattr(node_mod, "call_llm", _fail)
    out = await answer_node({"refined_query": "q", "documents": [], "domains": _INCIDENT})
    assert out["answerability_status"] == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE
    assert out["answerability_reason"]
    assert out["answer"] == FiveElements()


@pytest.mark.asyncio
async def test_answer_not_enough_by_llm_holds(monkeypatch):
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_ans_json("not_enough_evidence", ())))
    out = await answer_node({"refined_query": "q", "documents": [_doc()], "domains": _INCIDENT})
    assert out["answerability_status"] == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE
    assert out["answer"] == FiveElements()
    assert out["sources"] == []


@pytest.mark.asyncio
async def test_answer_partially_keeps_five(monkeypatch):
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_ans_json("partially_answerable", (0,))))
    out = await answer_node({"refined_query": "q", "documents": [_doc()], "domains": _INCIDENT})
    assert out["answerability_status"] == AnswerabilityStatus.PARTIALLY_ANSWERABLE
    assert out["answer"].situation != ""
    assert out["answerability_reason"]


@pytest.mark.asyncio
async def test_answer_citation_guard_demotes(monkeypatch):
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_ans_json("answerable", ())))
    out = await answer_node({"refined_query": "q", "documents": [_doc()], "domains": _INCIDENT})
    assert out["answerability_status"] == AnswerabilityStatus.PARTIALLY_ANSWERABLE
    assert out["sources"] == []
    assert out["answer"].situation != ""


@pytest.mark.asyncio
async def test_answer_consumes_gate_flags_conflicting(monkeypatch):
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_ans_json("answerable", (0,))))
    out = await answer_node(
        {"refined_query": "q", "documents": [_doc()], "domains": _INCIDENT, "gate_flags": GateFlags(conflicting=True)}
    )
    assert out["answerability_status"] == AnswerabilityStatus.CONFLICTING_EVIDENCE
    assert len(out["sources"]) == 1


@pytest.mark.asyncio
async def test_answer_parse_error_holds(monkeypatch):
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm("이건 JSON이 아님"))
    out = await answer_node({"refined_query": "q", "documents": [_doc()], "domains": _INCIDENT})
    assert out["answerability_status"] == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE
    assert "파싱" in out["answerability_reason"] or "생성 실패" in out["answerability_reason"]


@pytest.mark.asyncio
async def test_answer_llm_failure_holds(monkeypatch):
    async def _boom(*args, **kwargs):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(node_mod, "call_llm", _boom)
    out = await answer_node({"refined_query": "q", "documents": [_doc()], "domains": _INCIDENT})
    assert out["answerability_status"] == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE
    assert "LLM down" in out["error"]


@pytest.mark.asyncio
async def test_answer_coerces_list_solution(monkeypatch):
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
    out = await answer_node({"refined_query": "q", "documents": [_doc()], "domains": _INCIDENT})
    assert out["answerability_status"] == AnswerabilityStatus.ANSWERABLE
    assert "1. 첫 단계" in out["answer"].solution
    assert "2. 둘째 단계" in out["answer"].solution


@pytest.mark.asyncio
async def test_answer_source_mapping(monkeypatch):
    docs = [_doc("A"), _doc("B"), _doc("C")]
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_ans_json("answerable", (0, 2))))
    out = await answer_node({"refined_query": "q", "documents": docs, "domains": _INCIDENT})
    assert len(out["sources"]) == 2
    assert out["sources"][0].title == "A"
    assert out["sources"][1].title == "C"


def test_format_answer_unknown_status_coerces_but_keeps_five():
    payload = {
        "situation": "상황",
        "cause": "원인",
        "evidence": "근거",
        "solution": "해결",
        "infra_context": "맥락",
        "answerability_status": "totally_unknown_state",
        "source_indices": [0],
    }
    five, sources, status, parse_ok = format_answer(json.dumps(payload), [_doc()])
    assert parse_ok is True
    assert status == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE
    assert five.situation == "상황"
    assert len(sources) == 1


def test_format_freeform_unknown_status_coerces_but_keeps_text():
    # freeform도 동일: 알 수 없는 status는 NOT_ENOUGH로 매핑하되 answer_text는 보존
    payload = {"answer_text": "본문", "answerability_status": "weird", "source_indices": [0]}
    text, sources, status, parse_ok = format_freeform(json.dumps(payload), [_doc()])
    assert parse_ok is True
    assert status == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE
    assert text == "본문"
    assert len(sources) == 1


# ── decide_answerability 판단 경계 단위 (P1 점수·게이트 경로) ──
def test_decide_gate_conflicting_outdated():
    docs = [_doc()]
    assert decide_answerability(docs, gate=GateFlags(conflicting=True)) == AnswerabilityStatus.CONFLICTING_EVIDENCE
    assert decide_answerability(docs, gate=GateFlags(deprecated_only=True)) == AnswerabilityStatus.OUTDATED_EVIDENCE


def test_decide_p1_score_thresholds():
    docs = [_doc()]
    assert decide_answerability(docs, evidence_score=0.85) == AnswerabilityStatus.ANSWERABLE
    assert decide_answerability(docs, evidence_score=0.80) == AnswerabilityStatus.ANSWERABLE
    assert decide_answerability(docs, evidence_score=0.79) == AnswerabilityStatus.PARTIALLY_ANSWERABLE
    assert decide_answerability(docs, evidence_score=0.60) == AnswerabilityStatus.PARTIALLY_ANSWERABLE
    assert decide_answerability(docs, evidence_score=0.59) == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE
    assert decide_answerability(docs, evidence_score=0.40) == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE


def test_decide_floor_no_docs_overrides_llm():
    assert (
        decide_answerability([], llm_status=AnswerabilityStatus.ANSWERABLE) == AnswerabilityStatus.NOT_ENOUGH_EVIDENCE
    )


# ── Trust evidence_score 연결 + 미회수 버전 사유 (#108) ──────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overall", "expected"),
    [
        (0.85, AnswerabilityStatus.ANSWERABLE),
        (0.65, AnswerabilityStatus.PARTIALLY_ANSWERABLE),
        (0.30, AnswerabilityStatus.NOT_ENOUGH_EVIDENCE),
    ],
)
async def test_answer_uses_trust_overall_as_evidence_score(monkeypatch, overall, expected):
    from app.agents.state import TrustScore

    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_ans_json("answerable", (0,))))
    out = await answer_node(
        {"refined_query": "q", "documents": [_doc()], "domains": _INCIDENT, "trust_score": TrustScore(overall=overall)}
    )
    assert out["answerability_status"] == expected


@pytest.mark.asyncio
async def test_answer_caps_answerable_when_trust_has_no_good_topics(monkeypatch):
    """리랭커 fallback(raw=0)처럼 good topic이 없으면 overall이 높아도 answerable로 과신하지 않는다."""
    from app.agents.state import TrustScore

    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_ans_json("answerable", (0,))))
    out = await answer_node(
        {
            "refined_query": "q",
            "documents": [_doc()],
            "domains": _INCIDENT,
            "trust_score": TrustScore(overall=0.95, n_good_topics=0),
        }
    )
    assert out["answerability_status"] == AnswerabilityStatus.PARTIALLY_ANSWERABLE
    assert out["answer"].situation != ""  # 보류가 아니라 제한 답변
    assert "리랭커" in out["answerability_reason"]


@pytest.mark.asyncio
async def test_answer_gate_takes_priority_over_evidence_score(monkeypatch):
    from app.agents.state import TrustScore

    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_ans_json("answerable", (0,))))
    out = await answer_node(
        {
            "refined_query": "q",
            "documents": [_doc()],
            "domains": _INCIDENT,
            "trust_score": TrustScore(overall=0.95),
            "gate_flags": GateFlags(deprecated_only=True),
        }
    )
    assert out["answerability_status"] == AnswerabilityStatus.OUTDATED_EVIDENCE


@pytest.mark.asyncio
async def test_answer_partially_mentions_missing_versions(monkeypatch):
    from app.agents.state import TrustScore

    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_ans_json("answerable", (0,))))
    out = await answer_node(
        {
            "refined_query": "q",
            "documents": [_doc()],
            "domains": _INCIDENT,
            "trust_score": TrustScore(overall=0.65),
            "missing_versions": ["1.25"],
        }
    )
    assert out["answerability_status"] == AnswerabilityStatus.PARTIALLY_ANSWERABLE
    assert "1.25" in out["answerability_reason"]


@pytest.mark.asyncio
async def test_answer_sources_sorted_by_per_doc_evidence(monkeypatch):
    weak = SourceDocument(title="약한", content_snippet="w", per_doc_evidence=0.3)
    strong = SourceDocument(title="강한", content_snippet="s", per_doc_evidence=0.9)
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm(_ans_json("answerable", (0, 1))))
    out = await answer_node({"refined_query": "q", "documents": [weak, strong], "domains": _INCIDENT})
    assert [s.title for s in out["sources"]] == ["강한", "약한"]
