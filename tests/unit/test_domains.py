"""5도메인 단일 ontology — 문서 분류 프롬프트 생성 + 다른 정의 소스와의 정합 검증."""

from app.agents.state import Domain
from app.rag.classifier import DOMAIN_RULES
from app.rag.domains import DOMAIN_DEFINITIONS, DOMAIN_KEYS, domain_definition_block


def test_domain_keys_match_five():
    assert DOMAIN_KEYS == ("incident", "manual", "api_reference", "meeting_note", "planning")


def test_ontology_matches_other_definition_sources():
    # 단일 출처를 강제할 수는 없으므로(여러 곳에 정의 잔존), 최소한 키 집합 드리프트를 회귀로 막는다
    assert set(DOMAIN_KEYS) == {domain.value for domain in Domain}
    assert set(DOMAIN_KEYS) == set(DOMAIN_RULES)


def test_definition_block_differs_only_in_header_line():
    router = domain_definition_block("router").splitlines()
    document = domain_definition_block("document").splitlines()
    # 첫 줄(관점 헤더)만 다르고, 도메인 정의 라인은 완전히 동일해야 한다(드리프트 차단)
    assert router[0] != document[0]
    assert "질문이" in router[0]
    assert "문서가" in document[0]
    assert router[1:] == document[1:]
    for key in DOMAIN_KEYS:
        assert any(key in line for line in router[1:])


def test_definitions_have_boundary_where_corrected():
    by_key = {d.key: d for d in DOMAIN_DEFINITIONS}
    assert "manual" in by_key["api_reference"].boundary  # 사용법은 manual로 보냄
    assert "manual" in by_key["incident"].boundary  # 일반 점검은 manual
