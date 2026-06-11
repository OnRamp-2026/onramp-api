"""GoldenQuery.router_domains 파서 단위 테스트 (질의 의도 도메인 정답 + 출처 구분)."""

import pytest

from app.eval.dataset import GoldenQuery, _parse_router_domains


def _parse(row, *, domain, is_answerable=True):
    return _parse_router_domains(row, "q1", domain=domain, is_answerable=is_answerable)


def test_explicit_two_domains_loaded():
    # 1. 명시적 router_domains 2개 로딩 → explicit
    assert _parse({"router_domains": ["incident", "manual"]}, domain="incident") == (("incident", "manual"), "explicit")


def test_missing_field_falls_back_to_single_domain():
    # 2. 필드 없으면 domain 단일 fallback → source=fallback (공식 정답 아님)
    assert _parse({}, domain="manual") == (("manual",), "fallback")


def test_unanswerable_is_empty():
    # 3. unanswerable이면 빈 값 (명시값이 있어도 무시) → none
    assert _parse({"router_domains": ["incident", "manual"]}, domain=None, is_answerable=False) == ((), "none")


def test_more_than_two_rejected():
    # 4. 3개 이상 거부
    with pytest.raises(ValueError, match="최대 2개"):
        _parse({"router_domains": ["incident", "manual", "planning"]}, domain="incident")


def test_duplicate_rejected():
    # 5. 중복 거부
    with pytest.raises(ValueError, match="중복"):
        _parse({"router_domains": ["incident", "incident"]}, domain="incident")


def test_invalid_enum_rejected():
    # 6. 잘못된 enum 거부
    with pytest.raises(ValueError, match="알 수 없는 도메인"):
        _parse({"router_domains": ["bogus"]}, domain=None)


def test_empty_answerable_rejected():
    # 7. 빈 answerable router_domains 거부 (명시적 [])
    with pytest.raises(ValueError, match="빈 정답"):
        _parse({"router_domains": []}, domain="incident")


def test_order_preserved():
    # 8. 순서 보존 (우선순위)
    assert _parse({"router_domains": ["manual", "incident"]}, domain="manual") == (("manual", "incident"), "explicit")


def test_answerable_without_field_and_without_domain_is_empty_fallback():
    # answerable인데 router_domains도 domain도 없으면 무필터 fallback(빈, source=fallback) — ValueError 아님.
    # (기존 'domain=None=무필터' 계약 유지. 공식 지표는 explicit만 쓰므로 빈 fallback은 자동 제외)
    assert _parse({}, domain=None, is_answerable=True) == ((), "fallback")


def test_explicit_only_counts_for_official_metrics():
    # explicit만 공식 지표 대상. fallback은 검수 정답이 아니므로 제외된다.
    explicit = GoldenQuery(
        "m1", "q", None, True, (), router_domains=("incident", "manual"), router_domains_source="explicit"
    )
    fallback = GoldenQuery("d1", "q", "manual", True, (), router_domains=("manual",), router_domains_source="fallback")
    assert explicit.has_explicit_router_domains and explicit.is_multi_router_domain
    assert not fallback.has_explicit_router_domains
    assert not fallback.is_multi_router_domain  # 단일 fallback은 멀티로 세지 않음
