"""app/rag/lineage.py — Qdrant facet 파생 계보 조회 (#94)."""

from dataclasses import dataclass

import pytest

from app.config import Settings
from app.rag import lineage as lineage_mod
from app.rag.lineage import clear_lineage_cache, fetch_lineage, get_lineages


@dataclass
class _Hit:
    value: str


class _FakeFacetClient:
    """doc_key 필터 → product_version facet 응답 스텁. 호출 횟수를 기록한다."""

    def __init__(self, lineages: dict[str, list[str]]) -> None:
        self._lineages = lineages
        self.calls: list[str] = []

    def facet(self, collection_name: str, key: str, facet_filter, limit: int):  # noqa: ANN001
        assert key == "product_version"
        doc_key = facet_filter.must[0].match.value
        self.calls.append(doc_key)

        @dataclass
        class _Response:
            hits: list[_Hit]

        return _Response(hits=[_Hit(value=v) for v in self._lineages.get(doc_key, [])])


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_lineage_cache()
    yield
    clear_lineage_cache()


_SETTINGS = Settings(lineage_cache_ttl_seconds=300)


def test_fetch_lineage_aggregates_versions() -> None:
    client = _FakeFacetClient({"apache:content-negotiation": ["2.2", "2.4"]})
    assert fetch_lineage("apache:content-negotiation", client=client, settings=_SETTINGS) == frozenset({"2.2", "2.4"})


def test_fetch_lineage_empty_doc_key_skips_query() -> None:
    client = _FakeFacetClient({})
    assert fetch_lineage("", client=client, settings=_SETTINGS) == frozenset()
    assert client.calls == []


def test_get_lineages_batches_and_dedupes() -> None:
    client = _FakeFacetClient({"a:x": ["1.0"], "b:y": ["2.0", "3.0"]})
    result = get_lineages(["a:x", "b:y", "a:x", ""], client=client, settings=_SETTINGS)
    assert result["a:x"] == frozenset({"1.0"})
    assert result["b:y"] == frozenset({"2.0", "3.0"})
    assert result[""] == frozenset()
    assert client.calls == ["a:x", "b:y"]  # 중복·빈 키는 조회 안 함


def test_get_lineages_uses_ttl_cache() -> None:
    client = _FakeFacetClient({"a:x": ["1.0"]})
    get_lineages(["a:x"], client=client, settings=_SETTINGS)
    get_lineages(["a:x"], client=client, settings=_SETTINGS)
    assert client.calls == ["a:x"]  # 두 번째는 캐시 히트 — facet 1회만


def test_get_lineages_cache_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeFacetClient({"a:x": ["1.0"]})
    fake_now = [1000.0]
    monkeypatch.setattr(lineage_mod.time, "monotonic", lambda: fake_now[0])

    get_lineages(["a:x"], client=client, settings=_SETTINGS)
    fake_now[0] += _SETTINGS.lineage_cache_ttl_seconds + 1
    get_lineages(["a:x"], client=client, settings=_SETTINGS)
    assert client.calls == ["a:x", "a:x"]  # 만료 후 재조회


def test_get_lineages_ttl_zero_disables_cache() -> None:
    settings = Settings(lineage_cache_ttl_seconds=0)
    client = _FakeFacetClient({"a:x": ["1.0"]})
    get_lineages(["a:x"], client=client, settings=settings)
    get_lineages(["a:x"], client=client, settings=settings)
    assert client.calls == ["a:x", "a:x"]


def test_get_lineages_fetch_failure_falls_back_to_empty() -> None:
    """facet 장애는 빈 계보 폴백(미캐싱) — 보조 신호가 요청을 실패시키면 안 된다 (#108)."""

    class _BoomClient:
        def facet(self, *a, **kw):
            raise RuntimeError("Qdrant down")

    result = get_lineages(["a:x"], client=_BoomClient(), settings=_SETTINGS)
    assert result["a:x"] == frozenset()
    # 실패는 캐시되지 않음 — 복구 후 재조회 가능
    ok_client = _FakeFacetClient({"a:x": ["1.0"]})
    assert get_lineages(["a:x"], client=ok_client, settings=_SETTINGS)["a:x"] == frozenset({"1.0"})
