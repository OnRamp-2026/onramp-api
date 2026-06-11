"""라우터 예측 캐시 단위 테스트 (atomic write + stale 판정)."""

from app.eval.router_cache import CacheMeta, PredictionRecord, is_fresh, load_cache, write_cache

_META = CacheMeta(
    requested_model="",
    effective_provider="openai",
    llm_provider="openai",
    default_model="gpt-4o-mini",
    prompt_sha="abc123",
    schema_version="1",
)


def _record(qid="m003", query_sha="deadbeef0000", **over):
    base = dict(
        qid=qid,
        query_sha=query_sha,
        raw_predicted_domains=["incident", "manual"],
        predicted_domains=["incident", "manual"],
        confidence=0.87,
        use_case="검색",
        parse_ok=True,
        fallback_reason=None,
        low_conf_empty=False,
        requested_model=_META.requested_model,
        effective_provider=_META.effective_provider,
        llm_provider=_META.llm_provider,
        default_model=_META.default_model,
        prompt_sha=_META.prompt_sha,
        schema_version=_META.schema_version,
        commit_sha="c0ffee",
        created_at="2026-06-11T00:00:00+00:00",
    )
    base.update(over)
    return PredictionRecord(**base)


def test_write_is_atomic_and_roundtrips(tmp_path):
    path = tmp_path / "cache" / "router_predictions.jsonl"
    write_cache([_record(), _record(qid="m004", query_sha="aaaa11112222")], path)
    assert not path.with_suffix(path.suffix + ".tmp").exists()  # tmp 정리됨
    loaded = load_cache(path)
    assert set(loaded) == {"m003", "m004"}
    assert loaded["m003"]["predicted_domains"] == ["incident", "manual"]


def test_fresh_when_query_and_meta_match(tmp_path):
    loaded = _roundtrip(tmp_path, _record())
    assert is_fresh(loaded["m003"], query_sha="deadbeef0000", meta=_META)


def test_stale_when_query_changes(tmp_path):
    loaded = _roundtrip(tmp_path, _record())
    assert not is_fresh(loaded["m003"], query_sha="different0000", meta=_META)


def test_stale_when_model_or_contract_changes(tmp_path):
    loaded = _roundtrip(tmp_path, _record())
    changed_model = CacheMeta(**{**_META.__dict__, "default_model": "gpt-4o"})
    changed_schema = CacheMeta(**{**_META.__dict__, "schema_version": "2"})
    changed_prompt = CacheMeta(**{**_META.__dict__, "prompt_sha": "zzz999"})
    assert not is_fresh(loaded["m003"], query_sha="deadbeef0000", meta=changed_model)
    assert not is_fresh(loaded["m003"], query_sha="deadbeef0000", meta=changed_schema)
    assert not is_fresh(loaded["m003"], query_sha="deadbeef0000", meta=changed_prompt)


def test_repro_meta_not_part_of_stale_key(tmp_path):
    # commit_sha·created_at가 달라도 stale 키가 아니므로 여전히 신선해야 한다
    loaded = _roundtrip(tmp_path, _record(commit_sha="9999999", created_at="2099-01-01T00:00:00+00:00"))
    assert is_fresh(loaded["m003"], query_sha="deadbeef0000", meta=_META)


def test_missing_field_treated_as_stale():
    assert not is_fresh({"qid": "m003", "query_sha": "deadbeef0000"}, query_sha="deadbeef0000", meta=_META)


def _roundtrip(tmp_path, rec):
    path = tmp_path / "router_predictions.jsonl"
    write_cache([rec], path)
    return load_cache(path)
