"""문서 도메인 분류 dry-run 오케스트레이션 (Step 2, #49) — LLM mock, 파일 I/O는 tmp_path."""

from types import SimpleNamespace

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
    assert stats.classified == 1 and stats.reused == 0
    rec = records[0]
    assert rec["page_id"] == "p1" and rec["page_version"] == 1
    assert rec["primary_domain"] == "incident" and rec["adopted_domains"] == ["incident"]
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
    assert stats.reused == 1 and clf.calls == 0  # llm 결과는 재호출 없음
    assert records[0] == rec


async def test_run_dry_run_does_not_reuse_rule_fallback():
    # 일시 LLM 장애로 폴백된 결과는 다음 실행에서 LLM 재시도해야 한다
    clf = _FakeClassifier()
    page = _page()
    rec = build_record(page, _result("rule_fallback"), classifier_model="gpt-4o-mini")
    assert rec["review_status"] == "pending"
    _, stats = await run_dry_run([page], clf, existing={record_reuse_key(rec): rec})
    assert stats.classified == 1 and stats.reused == 0 and clf.calls == 1


async def test_run_dry_run_reuses_approved_even_if_fallback():
    # 사람이 승인(approved)한 결과는 source 무관하게 재사용
    clf = _FakeClassifier()
    page = _page()
    rec = build_record(page, _result("rule_fallback"), classifier_model="gpt-4o-mini")
    rec["review_status"] = "approved"
    _, stats = await run_dry_run([page], clf, existing={record_reuse_key(rec): rec})
    assert stats.reused == 1 and clf.calls == 0


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
    assert stats.classified == 1 and stats.reused == 0


async def test_run_dry_run_recalls_on_prompt_version_change():
    clf = _FakeClassifier()
    stale = build_record(_page(), _result(), classifier_model="gpt-4o-mini")
    stale["prompt_version"] = "0"  # 옛 프롬프트 버전 → 키 불일치
    _, stats = await run_dry_run([_page()], clf, existing={record_reuse_key(stale): stale})
    assert stats.classified == 1 and stats.reused == 0


async def test_run_dry_run_force_ignores_existing():
    clf = _FakeClassifier()
    page = _page()
    existing = {record_reuse_key(build_record(page, _result(), classifier_model="gpt-4o-mini")): {}}
    _, stats = await run_dry_run([page], clf, existing=existing, force=True)
    assert stats.classified == 1 and clf.calls == 1


async def test_run_dry_run_fallback_stays_pending():
    clf = _FakeClassifier(source="rule_fallback")
    records, stats = await run_dry_run([_page()], clf)
    assert stats.fallback == 1
    assert records[0]["classification_source"] == "rule_fallback"
    assert records[0]["review_status"] == "pending"


def test_write_and_load_roundtrip(tmp_path):
    page = _page()
    rec = build_record(page, _result(), classifier_model="gpt-4o-mini")
    path = tmp_path / "out.jsonl"
    write_jsonl(path, [rec])
    loaded = load_existing(path)
    assert loaded[record_reuse_key(rec)]["page_id"] == "p1"
