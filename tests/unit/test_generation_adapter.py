"""생성 평가 어댑터 단위 테스트 (그래프 stub — LLM/Qdrant 불필요)."""

import app.eval.generation_adapter as adapter_mod
from app.agents.state import AnswerabilityStatus, FiveElements, SourceDocument
from app.eval.generation_adapter import GenerationResult, flatten_answer, generate_for_eval


def _doc(snippet: str) -> SourceDocument:
    return SourceDocument(title="t", content_snippet=snippet)


def _stub_graph(monkeypatch, state: dict) -> None:
    async def _ainvoke(_initial: dict) -> dict:
        return state

    monkeypatch.setattr(adapter_mod.compiled_graph, "ainvoke", _ainvoke)


def test_flatten_answer_joins_nonempty_fields() -> None:
    five = FiveElements(situation="상황값", cause="", evidence="근거값", solution="", infra_context="맥락값")
    out = flatten_answer(five)
    assert "상황: 상황값" in out
    assert "근거: 근거값" in out
    assert "인프라 맥락: 맥락값" in out
    assert "원인:" not in out  # 빈 필드 제외


def test_flatten_answer_none_is_empty() -> None:
    assert flatten_answer(None) == ""


async def test_generate_extracts_answer_and_contexts(monkeypatch) -> None:
    state = {
        "answer": FiveElements(situation="EKS 장애", solution="롤백"),
        "documents": [_doc("문맥1"), _doc("문맥2"), _doc("")],
        "answerability_status": AnswerabilityStatus.ANSWERABLE,
    }
    _stub_graph(monkeypatch, state)

    result = await generate_for_eval("질문")
    assert isinstance(result, GenerationResult)
    assert "상황: EKS 장애" in result.answer_text
    assert result.retrieved_contexts == ["문맥1", "문맥2"]  # 빈 snippet 제외
    assert result.n_docs == 3
    assert result.answerability_status == "answerable"
    assert result.is_evaluable is True
    assert result.has_reference is False  # GT 미전달


async def test_generate_carries_reference(monkeypatch) -> None:
    state = {
        "answer": FiveElements(situation="x"),
        "documents": [_doc("문맥")],
        "answerability_status": AnswerabilityStatus.ANSWERABLE,
    }
    _stub_graph(monkeypatch, state)

    result = await generate_for_eval("질문", reference="정답 문장")
    assert result.reference == "정답 문장"
    assert result.has_reference is True


def test_has_reference_blank_is_false() -> None:
    assert GenerationResult(query="q", answer_text="a", reference="  ").has_reference is False
    assert GenerationResult(query="q", answer_text="a", reference=None).has_reference is False


async def test_generate_hold_is_not_evaluable(monkeypatch) -> None:
    # 보류(답변 비어있음) → judge 대상에서 제외돼야 한다
    state = {
        "answer": FiveElements(),
        "documents": [_doc("문맥")],
        "answerability_status": AnswerabilityStatus.NOT_ENOUGH_EVIDENCE,
    }
    _stub_graph(monkeypatch, state)

    result = await generate_for_eval("질문")
    assert result.answer_text == ""
    assert result.is_evaluable is False


async def test_generate_no_docs_is_not_evaluable(monkeypatch) -> None:
    state = {"answer": FiveElements(situation="x"), "documents": [], "answerability_status": "answerable"}
    _stub_graph(monkeypatch, state)

    result = await generate_for_eval("질문")
    assert result.retrieved_contexts == []
    assert result.is_evaluable is False
