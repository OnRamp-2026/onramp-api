"""문서 도메인 분류 스키마·채택 규칙 (P1, #49). LLM 없이 결정론 로직만 검증."""

import pytest
from pydantic import ValidationError

from app.rag.doc_domain_classifier import (
    DEFAULT_SECONDARY_THRESHOLD,
    MAX_SECONDARY,
    PageDomainClassification,
    adopt_domains,
    build_doc_classifier_system_prompt,
)


def _cls(primary, doms):
    return PageDomainClassification(primary_domain=primary, domains=doms)


def test_valid_classification():
    c = _cls("incident", [{"domain": "incident", "confidence": 0.9, "evidence_headings": ["원인 분석"]}])
    assert c.primary_domain == "incident"


def test_primary_must_match_domains_first():
    with pytest.raises(ValidationError):
        _cls("manual", [{"domain": "incident", "confidence": 0.9, "evidence_headings": ["x"]}])


def test_unknown_domain_rejected():
    with pytest.raises(ValidationError):
        _cls("incident", [{"domain": "incidentt", "confidence": 0.9, "evidence_headings": ["x"]}])


def test_duplicate_domain_rejected():
    with pytest.raises(ValidationError):
        _cls(
            "incident",
            [
                {"domain": "incident", "confidence": 0.9, "evidence_headings": ["a"]},
                {"domain": "incident", "confidence": 0.8, "evidence_headings": ["b"]},
            ],
        )


def test_empty_domains_rejected():
    with pytest.raises(ValidationError):
        _cls("incident", [])


def test_adopt_keeps_only_qualified_secondary():
    c = _cls(
        "incident",
        [
            {"domain": "incident", "confidence": 0.95, "evidence_headings": ["원인"]},
            {"domain": "manual", "confidence": 0.8, "evidence_headings": ["복구 절차"]},  # 채택
            {"domain": "api_reference", "confidence": 0.9, "evidence_headings": []},  # 근거 없음 → 탈락
            {"domain": "planning", "confidence": 0.3, "evidence_headings": ["설계"]},  # 저신뢰 → 탈락
        ],
    )
    assert adopt_domains(c) == ["incident", "manual"]


def test_adopt_threshold_boundary_and_primary_first():
    c = _cls(
        "manual",
        [
            {"domain": "manual", "confidence": 0.5, "evidence_headings": []},  # primary는 무조건 포함
            {"domain": "incident", "confidence": DEFAULT_SECONDARY_THRESHOLD, "evidence_headings": ["장애"]},
        ],
    )
    adopted = adopt_domains(c)
    assert adopted[0] == "manual"  # primary 항상 첫 값
    assert "incident" in adopted  # 임계값 경계(>=)는 채택


def test_adopt_caps_secondary():
    c = _cls(
        "incident",
        [
            {"domain": "incident", "confidence": 0.9, "evidence_headings": ["a"]},
            {"domain": "manual", "confidence": 0.9, "evidence_headings": ["b"]},
            {"domain": "api_reference", "confidence": 0.9, "evidence_headings": ["c"]},
            {"domain": "planning", "confidence": 0.9, "evidence_headings": ["d"]},
        ],
    )
    assert len(adopt_domains(c)) == 1 + MAX_SECONDARY


def test_doc_classifier_prompt_has_keys_and_schema():
    p = build_doc_classifier_system_prompt()
    assert "primary_domain" in p and "evidence_headings" in p
    assert "문서가" in p  # 문서 관점
    assert "api_reference" in p
