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


async def test_generate_reads_freeform_answer_text(monkeypatch) -> None:
    # #191 포맷 분기: incident 외 도메인은 freeform → state["answer_text"]에만 답이 있다.
    # FiveElements(answer)가 비어도 freeform 답변을 평가 대상으로 잡아야 한다(누락 방지).
    state = {
        "answer": FiveElements(),  # structured 비어있음
        "answer_text": "helm install로 배포합니다.",  # freeform 본문
        "documents": [_doc("문맥")],
        "answerability_status": AnswerabilityStatus.ANSWERABLE,
    }
    _stub_graph(monkeypatch, state)

    result = await generate_for_eval("질문")
    assert result.answer_text == "helm install로 배포합니다."
    assert result.is_evaluable is True


async def test_structured_answer_takes_precedence_over_freeform(monkeypatch) -> None:
    # structured(FiveElements)가 있으면 그걸 쓴다(freeform fallback은 비었을 때만).
    state = {
        "answer": FiveElements(situation="구조화 상황"),
        "answer_text": "이건 안 쓰여야 함",
        "documents": [_doc("문맥")],
        "answerability_status": AnswerabilityStatus.ANSWERABLE,
    }
    _stub_graph(monkeypatch, state)

    result = await generate_for_eval("질문")
    assert "상황: 구조화 상황" in result.answer_text
    assert "이건 안 쓰여야 함" not in result.answer_text


async def test_retrieved_contexts_use_parent_in_parent_mode(monkeypatch) -> None:
    # #212 faithfulness fix: parent-expanded면 retrieved_contexts가 LLM이 본 parent 본문이어야 한다
    # (child snippet으로 채점하면 parent 모드 faithfulness가 부당히 낮아진다).
    doc = SourceDocument(title="t", content_snippet="child snippet", parent_id="p1")
    state = {
        "answer": FiveElements(situation="x"),
        "documents": [doc],
        "answerability_status": AnswerabilityStatus.ANSWERABLE,
    }
    _stub_graph(monkeypatch, state)

    async def _fake_fetch(_documents):
        return {"p1": "PARENT 본문 전체"}  # parent-expanded에서 노드가 조회하는 것과 동일

    monkeypatch.setattr(adapter_mod, "_fetch_parent_contexts", _fake_fetch)

    result = await generate_for_eval("질문")
    assert result.retrieved_contexts == ["PARENT 본문 전체"]  # child snippet 아님


async def test_retrieved_contexts_child_only_uses_snippet(monkeypatch) -> None:
    # child-only(parent_contexts 없음)면 child snippet을 쓴다(기존 동작 유지).
    doc = SourceDocument(title="t", content_snippet="child snippet", parent_id="p1")
    state = {"answer": FiveElements(situation="x"), "documents": [doc], "answerability_status": "answerable"}
    _stub_graph(monkeypatch, state)

    async def _fake_fetch(_documents):
        return {}  # parent_context_enabled off

    monkeypatch.setattr(adapter_mod, "_fetch_parent_contexts", _fake_fetch)

    result = await generate_for_eval("질문")
    assert result.retrieved_contexts == ["child snippet"]


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


async def test_generate_captures_rerank_fallback_and_cost(monkeypatch) -> None:
    # #212: state의 rerank_fallback을 그대로 노출하고, 비용 필드는 안전한 기본값을 갖는다.
    state = {
        "answer": FiveElements(situation="x"),
        "documents": [_doc("문맥")],
        "answerability_status": AnswerabilityStatus.ANSWERABLE,
        "rerank_fallback": True,
    }
    _stub_graph(monkeypatch, state)

    result = await generate_for_eval("질문")
    assert result.rerank_fallback is True
    assert result.latency_s >= 0.0
    # stub 그래프는 call_llm을 호출하지 않으므로 token/호출수는 0이어야 한다(누산기 동작 확인).
    assert result.total_tokens == 0
    assert result.llm_calls == 0


async def test_generate_rerank_fallback_defaults_false(monkeypatch) -> None:
    state = {"answer": FiveElements(situation="x"), "documents": [_doc("c")], "answerability_status": "answerable"}
    _stub_graph(monkeypatch, state)

    result = await generate_for_eval("질문")
    assert result.rerank_fallback is False
