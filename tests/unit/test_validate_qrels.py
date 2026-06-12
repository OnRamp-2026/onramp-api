"""qrels 마이그레이션 검증 단위 테스트 (상태 분류 + scroll pagination, Qdrant 불필요)."""

from scripts.validate_qrels import _candidates, _classify, _page_id, _scroll_corpus


def test_page_id_extracts_prefix():
    assert _page_id("100_003") == "100"
    assert _page_id("abc_12_007") == "abc_12"  # 마지막 _ 앞
    assert _page_id("noidx") == "noidx"


def test_classify_all_states():
    existing = {"a_1", "a_2", "b_1"}
    assert _classify(("a_1", "a_2"), existing) == "intact"  # 전부 존재
    assert _classify(("a_1", "z_9"), existing) == "partial"  # 일부
    assert _classify(("z_9", "y_8"), existing) == "missing"  # 전부 없음
    assert _classify((), existing) == "empty"  # unanswerable


def test_candidates_groups_by_page_id():
    page_to = {"100": [{"chunk_id": "100_001", "preview": "x"}, {"chunk_id": "100_002", "preview": "y"}]}
    cand = _candidates(["100_009", "999_001"], page_to)
    assert [c["chunk_id"] for c in cand["100_009"]] == ["100_001", "100_002"]  # 같은 page_id 후보
    assert cand["999_001"] == []  # 해당 page 없음


# ── scroll pagination ──
class _Point:
    def __init__(self, payload):
        self.payload = payload


class _FakeClient:
    """offset이 None이 될 때까지 배치를 순서대로 반환하는 가짜 Qdrant 클라이언트."""

    def __init__(self, batches):
        self._batches = batches
        self._i = 0

    def scroll(self, **kwargs):
        batch = self._batches[self._i]
        self._i += 1
        return batch


def test_scroll_collects_all_pages_via_pagination():
    batches = [
        (
            [
                _Point({"chunk_id": "100_001", "content": "aaa", "page_id": "100"}),
                _Point({"chunk_id": "100_002", "content": "bbb"}),
            ],
            "offset1",
        ),  # page_id 없으면 chunk_id에서 유도
        ([_Point({"chunk_id": "200_001", "content": "ccc", "page_id": "200"})], None),  # offset None → 종료
    ]
    existing, page_to = _scroll_corpus(_FakeClient(batches), "onramp", preview_chars=2)
    assert existing == {"100_001", "100_002", "200_001"}  # 두 배치 모두 수집
    assert len(page_to["100"]) == 2 and len(page_to["200"]) == 1
    assert page_to["100"][0]["preview"] == "aa"  # preview_chars=2 절단


def test_scroll_skips_points_without_chunk_id():
    batches = [([_Point({"content": "no id"}), _Point({"chunk_id": "1_1", "content": "ok"})], None)]
    existing, _ = _scroll_corpus(_FakeClient(batches), "onramp")
    assert existing == {"1_1"}
