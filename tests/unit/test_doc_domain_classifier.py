"""문서 도메인 분류 스키마·채택 규칙 (P1, #49). LLM 없이 결정론 로직만 검증."""

import pytest
from pydantic import ValidationError

from app.middleware.error_handler import LLMError
from app.rag import doc_domain_classifier as mod
from app.rag.doc_domain_classifier import (
    DEFAULT_SECONDARY_THRESHOLD,
    MAX_SECONDARY,
    DocumentDomainClassifier,
    DomainEvidence,
    PageDomainClassification,
    adopt_domains,
    build_doc_classifier_system_prompt,
    rule_fallback_classification,
)


def _cls(primary, doms):
    return PageDomainClassification(primary_domain=primary, domains=doms)


def test_valid_classification():
    c = _cls("incident", [{"domain": "incident", "confidence": 0.9, "evidence_headings": ["원인 분석"]}])
    assert c.primary_domain == "incident"


def test_primary_absent_from_domains_rejected():
    # primary가 domains에 아예 없을 때만 실패 (LLM이 엉뚱한 primary를 낸 경우)
    with pytest.raises(ValidationError):
        _cls("manual", [{"domain": "incident", "confidence": 0.9, "evidence_headings": ["x"]}])


def test_primary_reordered_to_front_when_not_first():
    # LLM이 순서를 안 지켜도(primary가 뒤에 있어도) fallback 없이 맨 앞으로 재정렬
    c = _cls(
        "manual",
        [
            {"domain": "incident", "confidence": 0.9, "evidence_headings": ["a"]},
            {"domain": "manual", "confidence": 0.8, "evidence_headings": ["b"]},
        ],
    )
    assert c.domains[0].domain == "manual"  # 재정렬됨, 거부 아님
    assert {d.domain for d in c.domains} == {"manual", "incident"}


def test_unknown_domain_rejected():
    # DomainEvidence 단위로 고립 — primary 불일치 같은 다른 규칙이 끼지 않게
    with pytest.raises(ValidationError):
        DomainEvidence(domain="incidentt", confidence=0.9, evidence_headings=["x"])


def test_blank_evidence_headings_normalized():
    # 공백뿐인 heading은 정규화로 제거 → 근거 없음으로 secondary 탈락
    c = _cls(
        "incident",
        [
            {"domain": "incident", "confidence": 0.9, "evidence_headings": ["원인"]},
            {"domain": "manual", "confidence": 0.9, "evidence_headings": [" ", ""]},
        ],
    )
    assert c.domains[1].evidence_headings == []
    assert adopt_domains(c) == ["incident"]


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


def test_adopt_secondary_by_confidence_not_output_order():
    # LLM 출력 순서(앞 2개)가 아니라 confidence 내림차순 상위 MAX_SECONDARY개를 채택해야 한다
    c = _cls(
        "incident",
        [
            {"domain": "incident", "confidence": 0.99, "evidence_headings": ["장애"]},
            {"domain": "manual", "confidence": 0.61, "evidence_headings": ["절차"]},
            {"domain": "meeting_note", "confidence": 0.65, "evidence_headings": ["회의"]},
            {"domain": "api_reference", "confidence": 0.95, "evidence_headings": ["옵션"]},
        ],
    )
    # 앞에서 2개(manual, meeting_note)가 아니라 고신뢰(api_reference, meeting_note) 채택
    assert adopt_domains(c) == ["incident", "api_reference", "meeting_note"]


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


def _patch_llm(monkeypatch, *, returns=None, raises=None):
    async def _fake(system, user, **kwargs):
        if raises is not None:
            raise raises
        return returns

    monkeypatch.setattr(mod, "call_llm", _fake)


def _patch_llm_sequence(monkeypatch, responses):
    """호출마다 다음 응답을 돌려준다. 받은 user 프롬프트를 기록(교정 프롬프트 검증용)."""
    seen: list[str] = []
    it = iter(responses)

    async def _fake(system, user, **kwargs):
        seen.append(user)
        return next(it)

    monkeypatch.setattr(mod, "call_llm", _fake)
    return seen


async def test_classify_page_error_correcting_retry(monkeypatch):
    # 재현 버그: primary=incident인데 domains=[manual] (primary 부재) → 1차 검증 실패
    bad = '{"primary_domain": "incident", "domains": [{"domain": "manual", "confidence": 0.9, "evidence_headings": ["절차"]}]}'
    good = '{"primary_domain": "manual", "domains": [{"domain": "manual", "confidence": 0.9, "evidence_headings": ["절차"]}]}'
    seen = _patch_llm_sequence(monkeypatch, [bad, good])
    r = await DocumentDomainClassifier(max_retries=1).classify_page(page_title="설치 절차", content="...")
    assert r.source == "llm"  # 교정 재시도로 폴백 회피
    assert r.classification.primary_domain == "manual"
    assert len(seen) == 2
    assert "검증 오류" in seen[1]  # 2차 프롬프트에 오류 되먹임


async def test_classify_page_llm_success(monkeypatch):
    _patch_llm(
        monkeypatch,
        returns='{"primary_domain": "incident", "domains": ['
        '{"domain": "incident", "confidence": 0.95, "evidence_headings": ["원인 분석"]},'
        '{"domain": "manual", "confidence": 0.8, "evidence_headings": ["복구 절차"]}]}',
    )
    r = await DocumentDomainClassifier().classify_page(page_title="DB 장애 복구", content="...")
    assert r.source == "llm"
    assert r.classification.primary_domain == "incident"
    assert r.adopted_domains == ["incident", "manual"]


async def test_classify_page_llm_out_of_order_is_not_fallback(monkeypatch):
    # 재현된 버그: LLM이 primary를 domains 뒤에 둬도 폴백하지 말고 재정렬해 llm으로 처리
    _patch_llm(
        monkeypatch,
        returns='{"primary_domain": "manual", "domains": ['
        '{"domain": "incident", "confidence": 0.7, "evidence_headings": ["원인"]},'
        '{"domain": "manual", "confidence": 0.9, "evidence_headings": ["절차"]}]}',
    )
    r = await DocumentDomainClassifier().classify_page(page_title="설치 절차", content="...")
    assert r.source == "llm"  # rule_fallback 아님
    assert r.classification.domains[0].domain == "manual"  # 재정렬됨
    assert r.adopted_domains[0] == "manual"


async def test_classify_page_falls_back_on_invalid_json(monkeypatch):
    _patch_llm(monkeypatch, returns="not json at all")
    r = await DocumentDomainClassifier().classify_page(page_title="kubectl 설치 절차", content="helm 설치 매뉴얼")
    assert r.source == "rule_fallback"
    assert r.adopted_domains == [r.classification.primary_domain]  # 폴백은 단일 도메인


async def test_classify_page_falls_back_on_llm_error(monkeypatch):
    _patch_llm(monkeypatch, raises=LLMError("upstream down"))
    r = await DocumentDomainClassifier().classify_page(page_title="장애 postmortem", content="root cause 분석")
    assert r.source == "rule_fallback"
    assert r.classification.primary_domain == "incident"  # 키워드 규칙


def test_rule_fallback_picks_domain_by_keyword():
    c = rule_fallback_classification("회의록", "참석자 및 결정사항 정리")
    assert c.primary_domain == "meeting_note"
    assert c.domains[0].confidence == 0.0  # 근거 없음 신호


def test_heading_aware_sampling_preserves_front_middle_back():
    from app.rag.doc_domain_classifier import heading_aware_sample

    md = "# 앞 heading\n" + "a" * 3000 + "\n## 중간 heading\n" + "b" * 3000 + "\n### 뒤 heading\n" + "c" * 3000
    out = heading_aware_sample(md, max_chars=1200)
    # 단순 앞부분 절단이면 뒤 heading이 사라진다 → 모든 heading 보존 확인
    assert "# 앞 heading" in out
    assert "## 중간 heading" in out
    assert "### 뒤 heading" in out
    assert len(out) <= 1200
    # 각 섹션 본문이 일부씩 포함(앞 섹션이 예산을 독식하지 않음)
    assert "b" in out and "c" in out


def test_heading_aware_sampling_noop_when_short():
    from app.rag.doc_domain_classifier import heading_aware_sample

    md = "# 제목\n짧은 본문"
    assert heading_aware_sample(md, max_chars=1000) == md


def test_heading_aware_sampling_samples_when_headings_exceed_cap():
    from app.rag.doc_domain_classifier import heading_aware_sample

    # heading 합이 상한을 넘는 문서 → 앞·중간·뒤 균등 샘플(tail 절단이면 뒤 heading이 사라짐)
    md = "\n".join(f"# 헤딩 {i}" for i in range(100)) + "\n" + "x" * 500
    out = heading_aware_sample(md, max_chars=120)
    assert len(out) <= 120
    assert "# 헤딩 0" in out  # 앞 heading 보존
    assert "# 헤딩 99" in out  # 뒤 heading 보존(tail 절단 아님)
