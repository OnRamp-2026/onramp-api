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
    payload_domains,
)
from app.config import Settings


def test_payload_domains_multi_and_backward_compat():
    assert payload_domains({"domains": ["incident", "manual"]}) == ["incident", "manual"]
    assert payload_domains({"domains": "manual"}) == ["manual"]  # 문자열 단일값 → 글자 분해 금지(회귀)
    assert payload_domains({"domain": "manual"}) == ["manual"]  # 단일 domain 하위호환
    assert payload_domains({}) == []


def test_apply_domain_weight_string_domains_regression():
    """domains가 문자열로 들어와도 도메인 매칭이 정상 동작한다(list('manual') 분해 버그 방지)."""
    s = Settings()
    w = s.retriever_domain_match_weight
    assert apply_domain_weight(0.5, {"domains": "manual"}, "manual", s) == 0.5 + w


def test_apply_domain_weight_multidomain():
    s = Settings()
    w = s.retriever_domain_match_weight
    # router domain이 문서 domains 집합에 들면 가산
    assert apply_domain_weight(0.5, {"domains": ["incident", "manual"]}, "manual", s) == 0.5 + w
    assert apply_domain_weight(0.5, {"domains": ["incident", "manual"]}, "planning", s) == 0.5  # 불일치
    assert apply_domain_weight(0.5, {"domain": "manual"}, "manual", s) == 0.5 + w  # 단일 하위호환


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


def test_apply_domain_weight_additive_and_negative():
    """도메인 일치 시 가산 — 음수 점수여도 단조 증가, 불일치/None은 무가중."""
    settings = Settings()  # domain_match_weight 0.1
    w = settings.retriever_domain_match_weight
    assert apply_domain_weight(-0.5, {"domain": "manual"}, "manual", settings) == -0.5 + w
    assert apply_domain_weight(0.5, {"domain": "manual"}, "manual", settings) == 0.5 + w
    assert apply_domain_weight(-0.5, {"domain": "api_reference"}, "manual", settings) == -0.5  # 불일치
    assert apply_domain_weight(-0.5, {"domain": "manual"}, None, settings) == -0.5  # 신뢰 도메인 없음


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
