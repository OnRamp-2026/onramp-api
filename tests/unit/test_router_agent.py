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
            '{"use_case": "검색", "domain": "장애대응", "refined_query": "EKS Pod CrashLoop 해결", "confidence": 0.95}'
        ),
    )
    out = await route_node({"query": "EKS Pod CrashLoop 해결법"})
    assert out["use_case"] == UseCase.SEARCH
    assert out["domain"] == Domain.INCIDENT
    assert out["refined_query"] == "EKS Pod CrashLoop 해결"


@pytest.mark.asyncio
async def test_route_api_spec(monkeypatch):
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm(
            '{"use_case": "검색", "domain": "API명세", "refined_query": "결제 API 응답 필드", "confidence": 0.9}'
        ),
    )
    out = await route_node({"query": "결제 API 응답에 뭐가 오는지 알려줘"})
    assert out["domain"] == Domain.API_SPEC


@pytest.mark.asyncio
async def test_route_unanswerable(monkeypatch):
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm('{"use_case": "답변불가", "domain": "운영매뉴얼", "refined_query": "", "confidence": 0.99}'),
    )
    out = await route_node({"query": "오늘 날씨 어때?"})
    assert out["use_case"] == UseCase.UNANSWERABLE


@pytest.mark.asyncio
async def test_route_no_asset_case(monkeypatch):
    # Router는 SEARCH/UNANSWERABLE만 반환 — ASSET 없음
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm('{"use_case": "검색", "domain": "기획서", "refined_query": "기획 의도", "confidence": 0.8}'),
    )
    out = await route_node({"query": "이 기능 왜 만들었어?"})
    assert out["use_case"] in (UseCase.SEARCH, UseCase.UNANSWERABLE)


@pytest.mark.asyncio
async def test_route_low_confidence_fallback(monkeypatch):
    # confidence < 0.5 → domain만 OPS_MANUAL로 fallback (검색은 진행)
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm('{"use_case": "검색", "domain": "API명세", "refined_query": "x", "confidence": 0.3}'),
    )
    out = await route_node({"query": "애매한 질문"})
    assert out["use_case"] == UseCase.SEARCH
    assert out["domain"] == Domain.OPS_MANUAL


@pytest.mark.asyncio
async def test_route_parse_error_fallback(monkeypatch):
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm("이건 JSON이 아님"))
    out = await route_node({"query": "원본 질문"})
    assert out["use_case"] == UseCase.SEARCH
    assert out["domain"] == Domain.OPS_MANUAL
    assert out["refined_query"] == "원본 질문"  # 파싱 실패 시 원본 유지


@pytest.mark.asyncio
async def test_route_llm_failure_fallback(monkeypatch):
    async def _boom(*args, **kwargs):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(node_mod, "call_llm", _boom)
    out = await route_node({"query": "질문"})
    assert out["use_case"] == UseCase.SEARCH
    assert out["error"] != ""


@pytest.mark.asyncio
async def test_route_adds_trace(monkeypatch):
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm('{"use_case": "검색", "domain": "회의록", "refined_query": "회고 결정사항", "confidence": 0.88}'),
    )
    out = await route_node({"query": "회고 결정사항 정리해줘"})
    assert out["agent_trace"] == ["router"]
