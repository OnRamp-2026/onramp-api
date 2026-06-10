"""문서 도메인 분류 dry-run 오케스트레이션 (Step 2, #49) — LLM mock, 파일 I/O는 tmp_path."""

import json
from types import SimpleNamespace

import pytest

from app.rag.doc_domain_classifier import (
    DOC_CLASSIFIER_PROMPT_VERSION,
    ClassificationResult,
    DomainEvidence,
    PageDomainClassification,
)
from app.rag.doc_domain_dryrun import (
    DryRunPage,
    build_record,
    load_existing,
    merge_records,
    record_reuse_key,
    run_dry_run,
    write_jsonl,
)
from app.rag.domains import ONTOLOGY_VERSION


def _result(source="llm"):
    classification = PageDomainClassification(
        primary_domain="incident",
        domains=[DomainEvidence(domain="incident", confidence=0.9, evidence_headings=["원인"])],
    )
    return ClassificationResult(classification, ["incident"], source)


class _FakeClassifier:
    """call_llm 없이 동작 — classify_page 호출 횟수와 받은 content를 기록."""

    def __init__(self, model="gpt-4o-mini", source="llm"):
        self.settings = SimpleNamespace(classifier_model=model)
        self.calls = 0
        self.seen_content: list[str] = []
        self._source = source

    async def classify_page(self, *, page_title, content, secondary_threshold=0.6):
        self.calls += 1
        self.seen_content.append(content)
        return _result(self._source)


def _page(version=1):
    return DryRunPage(page_id="p1", version=version, title="DB 장애", masked_markdown="[MASKED] 본문")


async def test_run_dry_run_builds_pending_record():
    clf = _FakeClassifier()
    records, stats = await run_dry_run([_page()], clf)
    assert stats.classified == 1
    assert stats.reused == 0
    rec = records[0]
    assert rec["page_id"] == "p1"
    assert rec["page_version"] == 1
    assert rec["primary_domain"] == "incident"
    assert rec["adopted_domains"] == ["incident"]
    assert rec["classification_source"] == "llm"
    assert rec["review_status"] == "pending"  # 자동 승격 금지
    assert rec["prompt_version"] == DOC_CLASSIFIER_PROMPT_VERSION
    assert rec["ontology_version"] == ONTOLOGY_VERSION


async def test_run_dry_run_passes_only_masked_content():
    # 마스킹되지 않은 원문을 별도로 전달하지 않는 구조 — classifier는 page.masked_markdown만 받는다
    clf = _FakeClassifier()
    await run_dry_run([_page()], clf)
    assert clf.seen_content == ["[MASKED] 본문"]


async def test_run_dry_run_reuses_existing_llm_result():
    clf = _FakeClassifier()
    page = _page()
    rec = build_record(page, _result("llm"), classifier_model="gpt-4o-mini")
    records, stats = await run_dry_run([page], clf, existing={record_reuse_key(rec): rec})
    assert stats.reused == 1
    assert clf.calls == 0  # llm 결과는 재호출 없음
    assert records[0] == rec


async def test_run_dry_run_does_not_reuse_rule_fallback():
    # 일시 LLM 장애로 폴백된 결과는 다음 실행에서 LLM 재시도해야 한다
    clf = _FakeClassifier()
    page = _page()
    rec = build_record(page, _result("rule_fallback"), classifier_model="gpt-4o-mini")
    assert rec["review_status"] == "pending"
    _, stats = await run_dry_run([page], clf, existing={record_reuse_key(rec): rec})
    assert stats.classified == 1
    assert stats.reused == 0
    assert clf.calls == 1


async def test_run_dry_run_reuses_approved_even_if_fallback():
    # 사람이 승인(approved)한 결과는 source 무관하게 재사용
    clf = _FakeClassifier()
    page = _page()
    rec = build_record(page, _result("rule_fallback"), classifier_model="gpt-4o-mini")
    rec["review_status"] = "approved"
    _, stats = await run_dry_run([page], clf, existing={record_reuse_key(rec): rec})
    assert stats.reused == 1
    assert clf.calls == 0


def test_merge_records_preserves_pages_outside_this_run():
    # 기존 100개 중 30개만 재실행해도 나머지가 사라지면 안 된다(검수본 유실 방지)
    p1 = build_record(_page(version=1), _result(), classifier_model="m")
    p1["page_id"] = "keep-me"
    existing = {record_reuse_key(p1): p1}
    new = build_record(DryRunPage("p2", 1, "t", "x"), _result(), classifier_model="m")
    merged = merge_records(existing, [new])
    ids = {r["page_id"] for r in merged}
    assert ids == {"keep-me", "p2"}  # 기존 보존 + 신규 추가


async def test_run_dry_run_recalls_on_version_change():
    clf = _FakeClassifier()
    existing = {record_reuse_key(build_record(_page(version=1), _result(), classifier_model="gpt-4o-mini")): {}}
    # 같은 페이지지만 version 2 → 키 불일치 → 재호출
    _, stats = await run_dry_run([_page(version=2)], clf, existing=existing)
    assert stats.classified == 1
    assert stats.reused == 0


async def test_run_dry_run_recalls_on_prompt_version_change():
    clf = _FakeClassifier()
    stale = build_record(_page(), _result(), classifier_model="gpt-4o-mini")
    stale["prompt_version"] = "0"  # 옛 프롬프트 버전 → 키 불일치
    _, stats = await run_dry_run([_page()], clf, existing={record_reuse_key(stale): stale})
    assert stats.classified == 1
    assert stats.reused == 0


async def test_run_dry_run_force_ignores_existing():
    clf = _FakeClassifier()
    page = _page()
    existing = {record_reuse_key(build_record(page, _result(), classifier_model="gpt-4o-mini")): {}}
    _, stats = await run_dry_run([page], clf, existing=existing, force=True)
    assert stats.classified == 1
    assert clf.calls == 1


async def test_run_dry_run_fallback_stays_pending():
    clf = _FakeClassifier(source="rule_fallback")
    records, stats = await run_dry_run([_page()], clf)
    assert stats.fallback == 1
    assert records[0]["classification_source"] == "rule_fallback"
    assert records[0]["review_status"] == "pending"


def test_merge_records_one_row_per_page_id_across_version_change():
    # 같은 page_id의 과거 버전 레코드는 제거 → page_id당 1줄(Step 6 stale 승인본 오용 방지)
    old = build_record(_page(version=1), _result(), classifier_model="gpt-4o-mini")
    old["review_status"] = "approved"
    new = build_record(_page(version=2), _result(), classifier_model="gpt-4o-mini")
    merged = merge_records({record_reuse_key(old): old}, [new])
    p1_versions = [r["page_version"] for r in merged if r["page_id"] == "p1"]
    assert p1_versions == [2]  # v1 제거, v2만 남음


def test_merge_records_one_row_per_page_id_across_prompt_change():
    # 프롬프트 버전이 바뀌어도 같은 page_id는 1줄만 — 최신 스냅샷 보장
    old = build_record(_page(), _result(), classifier_model="gpt-4o-mini")
    old["prompt_version"] = "0"
    new = build_record(_page(), _result(), classifier_model="gpt-4o-mini")  # prompt_version 현재값
    merged = merge_records({record_reuse_key(old): old}, [new])
    p1 = [r for r in merged if r["page_id"] == "p1"]
    assert len(p1) == 1
    assert p1[0]["prompt_version"] != "0"  # 옛 프롬프트 결과 제거


def test_load_existing_reports_missing_required_field(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text('{"page_id": "p1"}\n', encoding="utf-8")  # 필수 필드 누락(유효 JSON)
    with pytest.raises(ValueError, match="필수 필드 누락"):
        load_existing(path)


def test_load_existing_reports_non_dict_line(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")  # dict 아닌 JSON → TypeError를 위치 에러로
    with pytest.raises(ValueError, match="손상"):
        load_existing(path)


def test_merge_records_dedupes_same_page_id_within_run():
    # 같은 실행에 같은 page_id가 2개 들어와도 마지막 1줄만 (1줄 보장)
    a = build_record(_page(version=1), _result(), classifier_model="m")
    b = build_record(_page(version=2), _result(), classifier_model="m")  # 같은 page_id p1
    merged = merge_records({}, [a, b])
    p1 = [r for r in merged if r["page_id"] == "p1"]
    assert len(p1) == 1
    assert p1[0]["page_version"] == 2  # 마지막 것


def test_load_existing_reports_corrupted_line(tmp_path):
    good = build_record(_page(), _result(), classifier_model="m")
    path = tmp_path / "out.jsonl"
    path.write_text(json.dumps(good, ensure_ascii=False) + "\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="파싱 실패"):
        load_existing(path)


def test_write_and_load_roundtrip(tmp_path):
    page = _page()
    rec = build_record(page, _result(), classifier_model="gpt-4o-mini")
    path = tmp_path / "out.jsonl"
    write_jsonl(path, [rec])
    loaded = load_existing(path)
    assert loaded[record_reuse_key(rec)]["page_id"] == "p1"
