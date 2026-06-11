import sys
import threading
import types

import pytest

from app.agents.retriever.rerank import (
    CrossEncoderReranker,
    OnnxCrossEncoderReranker,
    _recency_factor,
    apply_domain_weight,
    apply_metadata_weight,
)
from app.config import Settings


def test_apply_domain_weight_primary_and_secondary():
    """#61 반전: 문서 단일 domain이 질의 domains[0]이면 primary, domains[1:]이면 secondary 가산."""
    s = Settings()
    wp = s.domain_primary_weight  # 대표 도메인 가중
    ws = s.domain_secondary_weight  # 추가 도메인 가중
    # 문서 domain == 질의 domains[0] → primary
    assert apply_domain_weight(0.5, {"domain": "manual"}, ["manual"], s) == 0.5 + wp
    # 문서 domain ∈ domains[1:] → secondary
    assert apply_domain_weight(0.5, {"domain": "incident"}, ["manual", "incident"], s) == 0.5 + ws
    # 불일치 → 무가중
    assert apply_domain_weight(0.5, {"domain": "planning"}, ["manual", "incident"], s) == 0.5
    # 빈 질의 도메인 → 무가중
    assert apply_domain_weight(0.5, {"domain": "manual"}, [], s) == 0.5
    assert apply_domain_weight(0.5, {"domain": "manual"}, None, s) == 0.5
    # 문서에 domain 없음 → 무가중
    assert apply_domain_weight(0.5, {}, ["manual"], s) == 0.5


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


def test_onnx_rerank_applies_sigmoid_to_match_torch_score_contract():
    """ONNX 경로는 순수 numpy sigmoid로 점수를 내며, torch 백엔드와 동일한 [0,1] 점수 계약을 따른다."""
    import numpy as np

    logits = np.array([[-1.1073], [1.0]], dtype=np.float32)

    class _Tokenizer:
        def __call__(self, *args, **kwargs):
            return {"input_ids": np.zeros((2, 1), dtype=np.int64)}

    class _Session:
        def get_inputs(self):
            return [types.SimpleNamespace(name="input_ids")]

        def run(self, _outputs, _inputs):
            return [logits]

    reranker = OnnxCrossEncoderReranker(settings=Settings())
    reranker._tokenizer = _Tokenizer()
    reranker._session = _Session()  # lazy 로드 우회
    reranker._input_names = {"input_ids"}

    out = reranker.rerank("q", [("a", {"id": 1}), ("b", {"id": 2})])

    expected = (1.0 / (1.0 + np.exp(-logits))).reshape(-1).tolist()
    assert [payload["id"] for _, payload in out] == [2, 1]
    assert [score for score, _ in out] == pytest.approx([expected[1], expected[0]])
    assert all(0.0 <= score <= 1.0 for score, _ in out)


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


def test_apply_metadata_weight_additive_on_negative():
    """음수 점수에서도 최신성 가산은 점수를 올린다 (곱셈이면 더 낮아지던 버그)."""
    settings = Settings()
    assert apply_metadata_weight(-0.5, {"last_modified": "2026-06-01T00:00:00Z"}, settings) > -0.5
    assert apply_metadata_weight(-0.5, {}, settings) == -0.5  # 날짜 없으면 무가중


def test_apply_domain_weight_primary_gt_secondary_and_negative():
    """primary > secondary(대표 우선), 음수 점수에서도 단조 증가."""
    s = Settings()
    assert s.domain_primary_weight > s.domain_secondary_weight
    assert apply_domain_weight(-0.5, {"domain": "manual"}, ["manual"], s) == -0.5 + s.domain_primary_weight
    assert (
        apply_domain_weight(-0.5, {"domain": "manual"}, ["incident", "manual"], s) == -0.5 + s.domain_secondary_weight
    )


def test_lazy_load_thread_safe(monkeypatch):
    """동시 cold-start 8스레드에도 CrossEncoder는 1회만 생성된다."""
    instances = []

    class _CountingCrossEncoder:
        def __init__(self, *args: object, **kwargs: object) -> None:
            instances.append(1)

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = _CountingCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    reranker = CrossEncoderReranker(settings=Settings())
    barrier = threading.Barrier(8)

    def hit() -> None:
        barrier.wait()  # 동시 진입 극대화
        _ = reranker.model

    threads = [threading.Thread(target=hit) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(instances) == 1
