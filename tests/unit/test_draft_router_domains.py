"""검수표 생성기 — 재실행 시 사람 검수 결과 보존 회귀 테스트."""

import json

from app.eval.dataset import GoldenQuery
from app.eval.router_cache import CacheMeta
from scripts.draft_router_domains import _load_existing, _row

_META = CacheMeta("", "openai", "openai", "gpt-4o-mini", "abc", "1")


def _golden(qid="m003"):
    return GoldenQuery(qid, "장애 원인과 복구 절차", "incident", True, (), gold_domains=("incident", "manual"))


def _approved(query_sha):
    return {
        "qid": "m003",
        "query_sha": query_sha,
        "reviewed_router_domains": ["incident", "manual"],
        "review_status": "approved",
        "reviewer": "jihong",
        "reviewed_at": "2026-06-11T10:00:00+00:00",
    }


def _fresh_rec(qid="m003", query_sha="qs"):
    """is_fresh를 통과하는 캐시 레코드(meta가 _META와 일치)."""
    return {
        "qid": qid,
        "query_sha": query_sha,
        "raw_predicted_domains": ["incident", "manual"],
        "predicted_domains": ["incident", "manual"],
        "requested_model": "",
        "effective_provider": "openai",
        "llm_provider": "openai",
        "default_model": "gpt-4o-mini",
        "prompt_sha": "abc",
        "schema_version": "1",
    }


def test_reviewed_fields_preserved_when_query_unchanged():
    # query_sha 일치 → reviewed_*·status·reviewer·reviewed_at 보존
    row = _row(_golden(), {}, _approved("samesha"), query_sha="samesha", meta=_META, blind=False)
    assert row["reviewed_router_domains"] == ["incident", "manual"]
    assert row["review_status"] == "approved"
    assert row["reviewer"] == "jihong"
    assert row["reviewed_at"] == "2026-06-11T10:00:00+00:00"
    assert row["query_sha"] == "samesha"
    # 제안 관련 필드는 매 실행 갱신(캐시 비었으니 제안 없음)
    assert row["suggestion_source"] == "none"
    assert row["proposed_router_domains"] is None


def test_reviewed_fields_invalidated_when_query_changes():
    # query_sha 불일치(질문 문구 변경) → 옛 approved 라벨을 버리고 pending으로 재검수 강제
    row = _row(_golden(), {}, _approved("oldsha"), query_sha="newsha", meta=_META, blind=False)
    assert row["review_status"] == "pending"
    assert row["reviewed_router_domains"] is None
    assert row["reviewer"] is None and row["reviewed_at"] is None
    assert row["query_sha"] == "newsha"


def test_defaults_to_pending_when_no_existing():
    row = _row(_golden(), {}, None, query_sha="x", meta=_META, blind=False)
    assert row["review_status"] == "pending"
    assert row["reviewed_router_domains"] is None
    assert row["reviewer"] is None and row["reviewed_at"] is None


def test_blind_omits_proposal_from_review_row():
    # --blind: 중요 문항(m0xx=multi)의 제안을 검수표 행에서 **제거**(같은 행에 두면 blind가 무의미)
    cache = {"m003": _fresh_rec("m003", "qs")}
    blinded = _row(_golden("m003"), cache, None, query_sha="qs", meta=_META, blind=True)
    assert blinded["suggestion_source"] == "blinded"
    assert blinded["proposed_router_domains"] is None
    assert "blinded_suggestion" not in blinded  # 행에 제안이 남으면 안 됨(sidecar로만)
    # blind 아니면 **raw**(게이팅 전) 제안을 노출 — 저신뢰로 비워지기 전 LLM 실제 분류
    shown = _row(_golden("m003"), cache, None, query_sha="qs", meta=_META, blind=False)
    assert shown["suggestion_source"] == "router_prediction"
    assert shown["proposed_router_domains"] == ["incident", "manual"]


def test_proposal_uses_raw_not_gated():
    # 저신뢰로 게이팅돼 predicted_domains가 비어도, 제안은 raw_predicted_domains를 쓴다
    rec = _fresh_rec("d001", "qs")
    rec["raw_predicted_domains"] = ["incident"]
    rec["predicted_domains"] = []  # 게이팅 후 빔
    row = _row(_golden("d001"), {"d001": rec}, None, query_sha="qs", meta=_META, blind=False)
    assert row["proposed_router_domains"] == ["incident"]  # gated([])가 아니라 raw
    assert row["suggestion_source"] == "router_prediction"


def test_load_existing_roundtrip(tmp_path):
    path = tmp_path / "router_domains_review.jsonl"
    rows = [{"qid": "m003", "review_status": "approved"}, {"qid": "c001", "review_status": "pending"}]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    loaded = _load_existing(path)
    assert set(loaded) == {"m003", "c001"}
    assert loaded["m003"]["review_status"] == "approved"


def test_load_existing_missing_file_returns_empty(tmp_path):
    assert _load_existing(tmp_path / "nope.jsonl") == {}
