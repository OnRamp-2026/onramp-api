"""Router Agent 단위 테스트 (LLM mock 사용)."""

import pytest

from app.agents.router import node as node_mod
from app.agents.router.node import route_node
from app.agents.state import Domain, UseCase


def _mock_llm(response: str):
    async def _call(*args, **kwargs):
        return response

    return _call


@pytest.mark.asyncio
async def test_route_incident(monkeypatch):
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm(
            '{"use_case": "검색", "domains": ["incident"], "refined_query": "EKS Pod CrashLoop 해결", "confidence": 0.95}'
        ),
    )
    out = await route_node({"query": "EKS Pod CrashLoop 해결법"})
    assert out["use_case"] == UseCase.SEARCH
    assert out["domains"] == [Domain.INCIDENT]
    assert out["domain"] == Domain.INCIDENT  # 하위호환 = domains[0]
    assert out["refined_query"] == "EKS Pod CrashLoop 해결"


@pytest.mark.asyncio
async def test_route_multidomain(monkeypatch):
    """질의가 두 도메인을 요구하면 순서 있는 domains, domain은 domains[0] 파생."""
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm(
            '{"use_case": "검색", "domains": ["incident", "manual"], '
            '"refined_query": "장애 원인과 복구 절차", "confidence": 0.9}'
        ),
    )
    out = await route_node({"query": "장애 원인이랑 복구 절차 알려줘"})
    assert out["domains"] == [Domain.INCIDENT, Domain.MANUAL]
    assert out["domain"] == Domain.INCIDENT  # 대표 = domains[0]


@pytest.mark.asyncio
async def test_route_api_spec(monkeypatch):
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm(
            '{"use_case": "검색", "domains": ["api_reference"], "refined_query": "결제 API 응답 필드", "confidence": 0.9}'
        ),
    )
    out = await route_node({"query": "결제 API 응답에 뭐가 오는지 알려줘"})
    assert out["domain"] == Domain.API_REFERENCE


@pytest.mark.asyncio
async def test_route_unanswerable(monkeypatch):
    # LLM이 refined_query를 잘못 채워도 노드가 UNANSWERABLE이면 강제로 비운다
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm(
            '{"use_case": "답변불가", "domains": ["manual"], "refined_query": "잘못 채운 값", "confidence": 0.99}'
        ),
    )
    out = await route_node({"query": "오늘 날씨 어때?"})
    assert out["use_case"] == UseCase.UNANSWERABLE
    assert out["refined_query"] == ""  # 노드에서 계약 보장
    assert out["answerability_reason"]  # 차단 사유 안내 메시지 채움


@pytest.mark.asyncio
async def test_route_no_asset_case(monkeypatch):
    # Router는 SEARCH/UNANSWERABLE만 반환 — ASSET 없음
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm('{"use_case": "검색", "domains": ["planning"], "refined_query": "기획 의도", "confidence": 0.8}'),
    )
    out = await route_node({"query": "이 기능 왜 만들었어?"})
    assert out["use_case"] in (UseCase.SEARCH, UseCase.UNANSWERABLE)


@pytest.mark.asyncio
async def test_route_low_confidence_no_filter(monkeypatch):
    # confidence < 0.5 → 도메인 신뢰 불가 → None(무필터). manual로 가두지 않는다 (검색은 진행)
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm('{"use_case": "검색", "domains": ["api_reference"], "refined_query": "x", "confidence": 0.3}'),
    )
    out = await route_node({"query": "애매한 질문"})
    assert out["use_case"] == UseCase.SEARCH
    assert out["domain"] is None


@pytest.mark.asyncio
async def test_route_confidence_boundary_keeps_domain(monkeypatch):
    # confidence == 0.5 (임계값) → domain 유지 (>= 비교이므로 fallback 아님)
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm('{"use_case": "검색", "domains": ["api_reference"], "refined_query": "x", "confidence": 0.5}'),
    )
    out = await route_node({"query": "경계값 질문"})
    assert out["domain"] == Domain.API_REFERENCE


@pytest.mark.asyncio
async def test_route_parse_error_fallback(monkeypatch):
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm("이건 JSON이 아님"))
    out = await route_node({"query": "원본 질문"})
    assert out["use_case"] == UseCase.SEARCH
    assert out["domain"] is None  # 신뢰할 도메인 없음 → 무필터
    assert out["refined_query"] == "원본 질문"  # 파싱 실패 시 원본 유지


@pytest.mark.asyncio
async def test_route_llm_failure_fallback(monkeypatch):
    async def _boom(*args, **kwargs):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(node_mod, "call_llm", _boom)
    out = await route_node({"query": "질문"})
    assert out["use_case"] == UseCase.SEARCH
    assert out["domain"] is None  # 신뢰할 도메인 없음 → 무필터
    assert out["refined_query"] == "질문"  # 원본 질문 유지
    assert "LLM down" in out["error"]  # 예외 메시지가 error에 기록


@pytest.mark.asyncio
async def test_route_adds_trace(monkeypatch):
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm(
            '{"use_case": "검색", "domains": ["meeting_note"], "refined_query": "회고 결정사항", "confidence": 0.88}'
        ),
    )
    out = await route_node({"query": "회고 결정사항 정리해줘"})
    assert out["agent_trace"] == ["router"]
