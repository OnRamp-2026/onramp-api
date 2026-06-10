"""5도메인 단일 ontology — 라우터/문서 분류가 같은 정의를 공유하는지 검증."""

from app.agents.router.prompts import ROUTER_SYSTEM_PROMPT
from app.rag.domains import DOMAIN_DEFINITIONS, DOMAIN_KEYS, domain_definition_block


def test_domain_keys_match_five():
    assert DOMAIN_KEYS == ("incident", "manual", "api_reference", "meeting_note", "planning")


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


def test_router_prompt_built_from_shared_ontology():
    # 라우터 프롬프트가 ontology 정의(경계 보정 포함)를 그대로 싣는다 — 드리프트 방지
    assert "정확한 문법" in ROUTER_SYSTEM_PROMPT  # api_reference 보정 정의
    for key in DOMAIN_KEYS:
        assert key in ROUTER_SYSTEM_PROMPT
    # few-shot JSON 중괄호가 깨지지 않고 보존됐는지
    assert '{"use_case": "검색", "domain": "incident"' in ROUTER_SYSTEM_PROMPT


def test_definitions_have_boundary_where_corrected():
    by_key = {d.key: d for d in DOMAIN_DEFINITIONS}
    assert "manual" in by_key["api_reference"].boundary  # 사용법은 manual로 보냄
    assert "manual" in by_key["incident"].boundary  # 일반 점검은 manual
