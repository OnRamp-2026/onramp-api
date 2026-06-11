"""classify_query 진단값 + route_node 매핑 동일성(추출 회귀) 테스트."""

import pytest

from app.agents.router import node as node_mod
from app.agents.router.node import classify_query, route_node
from app.agents.state import Domain, UseCase


def _mock_llm(response: str):
    async def _call(*args, **kwargs):
        return response

    return _call


@pytest.mark.asyncio
async def test_classify_normal_search(monkeypatch):
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm(
            '{"use_case": "검색", "domains": ["incident", "manual"], "refined_query": "복구", "confidence": 0.9}'
        ),
    )
    diag = await classify_query("장애 복구")
    assert diag.use_case == UseCase.SEARCH
    assert diag.domains == [Domain.INCIDENT, Domain.MANUAL]
    assert diag.raw_domains == [Domain.INCIDENT, Domain.MANUAL]
    assert diag.confidence == 0.9
    assert diag.parse_ok is True and diag.fallback_reason is None


@pytest.mark.asyncio
async def test_classify_low_confidence_keeps_raw_but_gates_domains(monkeypatch):
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm('{"use_case": "검색", "domains": ["api_reference"], "refined_query": "x", "confidence": 0.3}'),
    )
    diag = await classify_query("애매")
    assert diag.domains == []  # 게이팅 후 비움
    assert diag.raw_domains == [Domain.API_REFERENCE]  # 원본 보존(저신뢰 구분용)
    assert diag.confidence == 0.3 and diag.parse_ok is True


@pytest.mark.asyncio
async def test_classify_parse_error(monkeypatch):
    monkeypatch.setattr(node_mod, "call_llm", _mock_llm("이건 JSON이 아님"))
    diag = await classify_query("원본")
    assert diag.fallback_reason == "parse_error"
    assert diag.confidence is None  # 실패는 0.0이 아니라 None (ECE 왜곡 방지)
    assert diag.parse_ok is False and diag.domains == []


@pytest.mark.asyncio
async def test_classify_llm_error(monkeypatch):
    async def _boom(*args, **kwargs):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(node_mod, "call_llm", _boom)
    diag = await classify_query("질문")
    assert diag.fallback_reason == "llm_error"
    assert diag.confidence is None
    assert diag.error and "LLM down" in diag.error


@pytest.mark.asyncio
async def test_route_node_maps_classify_faithfully(monkeypatch):
    """route_node 출력이 classify_query 결과의 충실한 매핑인지(추출 전후 동작 동일) 확인."""
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm(
            '{"use_case": "검색", "domains": ["incident", "manual"], "refined_query": "복구", "confidence": 0.9}'
        ),
    )
    diag = await classify_query("장애 복구")
    out = await route_node({"query": "장애 복구"})
    assert out["use_case"] == diag.use_case
    assert out["domains"] == diag.domains
    assert out["domain"] == diag.domains[0]  # 하위호환 파생
    assert out["refined_query"] == diag.refined_query
    assert out["agent_trace"] == ["router"]
