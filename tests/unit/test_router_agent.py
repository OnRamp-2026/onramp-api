"""Router Agent 단위 테스트 (LLM mock 사용)."""

import pytest
from pydantic import ValidationError

from app.agents.router import node as node_mod
from app.agents.router.node import route_node
from app.agents.router.schema import RouterOutput
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
async def test_route_unanswerable_with_duplicate_domains_still_blocks(monkeypatch):
    """UNANSWERABLE + 중복 domains여도 검증 실패(→SEARCH fallback)가 아니라 차단 유지(domains=[])."""
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm('{"use_case": "답변불가", "domains": ["manual", "manual"], "refined_query": "x", "confidence": 0.9}'),
    )
    out = await route_node({"query": "오늘 날씨 어때?"})
    assert out["use_case"] == UseCase.UNANSWERABLE  # SEARCH로 잘못 빠지지 않음
    assert out["domains"] == []
    assert out["domain"] is None


@pytest.mark.asyncio
async def test_route_unanswerable_with_too_many_domains_not_searched(monkeypatch):
    """UNANSWERABLE + domains 3개(max_length 초과)여도 필드검증 전 비워져 SEARCH로 새지 않는다.

    회귀: mode="after"만 있으면 max_length=2 위반이 먼저 ValidationError→SEARCH fallback돼
    답변불가 질문이 검색으로 전환되는 안전성 버그가 났다.
    """
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm(
            '{"use_case": "답변불가", "domains": ["manual", "incident", "planning"], '
            '"refined_query": "x", "confidence": 0.9}'
        ),
    )
    out = await route_node({"query": "오늘 점심 뭐 먹지?"})
    assert out["use_case"] == UseCase.UNANSWERABLE  # SEARCH로 잘못 빠지지 않음
    assert out["domains"] == []
    assert out["domain"] is None


@pytest.mark.asyncio
async def test_route_unanswerable_with_invalid_domain_not_searched(monkeypatch):
    """UNANSWERABLE + 잘못된 domain 문자열(enum 위반)이어도 비워져 SEARCH로 새지 않는다."""
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm('{"use_case": "답변불가", "domains": ["bogus"], "refined_query": "x", "confidence": 0.9}'),
    )
    out = await route_node({"query": "주식 추천해줘"})
    assert out["use_case"] == UseCase.UNANSWERABLE  # SEARCH로 잘못 빠지지 않음
    assert out["domains"] == []
    assert out["domain"] is None


def test_schema_search_rejects_too_many_domains():
    """SEARCH + domains 3개(max_length 초과) → ValidationError 유지."""
    with pytest.raises(ValidationError):
        RouterOutput.model_validate_json(
            '{"use_case": "검색", "domains": ["manual", "incident", "planning"], '
            '"refined_query": "x", "confidence": 0.9}'
        )


def test_schema_search_rejects_invalid_domain():
    """SEARCH + 잘못된 domain 문자열(enum 위반) → ValidationError 유지."""
    with pytest.raises(ValidationError):
        RouterOutput.model_validate_json(
            '{"use_case": "검색", "domains": ["bogus"], "refined_query": "x", "confidence": 0.9}'
        )


def test_schema_search_rejects_empty_domains():
    """SEARCH + domains=[] → ValidationError 유지 (최소 1개 필요)."""
    with pytest.raises(ValidationError):
        RouterOutput.model_validate_json('{"use_case": "검색", "domains": [], "refined_query": "x", "confidence": 0.9}')


def test_schema_rejects_invalid_use_case():
    """use_case 자체가 잘못된 값이면 정상적으로 ValidationError (before validator가 가로채지 않음)."""
    with pytest.raises(ValidationError):
        RouterOutput.model_validate_json(
            '{"use_case": "헛소리", "domains": ["manual"], "refined_query": "x", "confidence": 0.9}'
        )


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


# ── target_versions 추출 (#108) ──────────────────────────────────────


async def test_route_extracts_target_versions(monkeypatch):
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm(
            '{"use_case": "검색", "domains": ["manual"], "refined_query": "k8s 1.25 1.33 차이",'
            ' "confidence": 0.9, "target_versions": ["1.25", "1.33"]}'
        ),
    )
    out = await route_node({"query": "k8s 1.25에서 1.33으로 올리면?"})
    assert out["target_versions"] == ["1.25", "1.33"]


async def test_route_filters_non_numeric_version_tokens(monkeypatch):
    """LLM이 'latest' 류 토큰을 넣으면 방어 필터가 제거한다 — '최신'은 currency 모드가 정답."""
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm(
            '{"use_case": "검색", "domains": ["manual"], "refined_query": "q",'
            ' "confidence": 0.9, "target_versions": ["latest", "v1.33", "최신"]}'
        ),
    )
    out = await route_node({"query": "최신이랑 1.33"})
    assert out["target_versions"] == ["v1.33"]


async def test_route_missing_target_versions_key_defaults_empty(monkeypatch):
    """LLM이 target_versions 키를 빠뜨려도 파싱 실패 없이 빈 리스트."""
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm('{"use_case": "검색", "domains": ["manual"], "refined_query": "q", "confidence": 0.9}'),
    )
    out = await route_node({"query": "버전 없는 질문"})
    assert out["target_versions"] == []


async def test_route_unanswerable_clears_target_versions(monkeypatch):
    monkeypatch.setattr(
        node_mod,
        "call_llm",
        _mock_llm(
            '{"use_case": "답변불가", "domains": [], "refined_query": "",'
            ' "confidence": 0.95, "target_versions": ["1.25"]}'
        ),
    )
    out = await route_node({"query": "점심 뭐 먹지 1.25"})
    assert out["target_versions"] == []
