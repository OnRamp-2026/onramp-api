from app.agents.retriever.rerank import CrossEncoderReranker, _recency_factor, apply_metadata_weight
from app.config import Settings


class _FakeModel:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs):
        return self.scores


def test_rerank_sorts_desc():
    reranker = CrossEncoderReranker(settings=Settings())
    reranker._model = _FakeModel([0.1, 0.9, 0.5])  # lazy 로드 우회
    out = reranker.rerank("q", [("a", {"id": 1}), ("b", {"id": 2}), ("c", {"id": 3})])
    assert [p["id"] for _, p in out] == [2, 3, 1]


def test_rerank_empty():
    assert CrossEncoderReranker(settings=Settings()).rerank("q", []) == []


def test_recency_factor_fresh_gt_old_and_safe():
    fresh = _recency_factor("2026-05-30T00:00:00Z", 180)
    old = _recency_factor("2020-01-01T00:00:00Z", 180)
    assert fresh > old
    assert _recency_factor("", 180) == 0.0
    assert _recency_factor("not-a-date", 180) == 0.0


def test_apply_metadata_weight_bounded():
    settings = Settings()  # recency_weight 0.1
    weighted = apply_metadata_weight(1.0, {"last_modified": "2026-06-01T00:00:00Z"}, settings)
    assert 1.0 <= weighted <= 1.1 + 1e-9
    assert apply_metadata_weight(1.0, {}, settings) == 1.0  # 날짜 없으면 무가중
