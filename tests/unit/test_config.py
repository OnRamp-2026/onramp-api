"""Settings 범위 검증 — 환경변수로 오염된 도메인 보정 값을 차단한다."""

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_domain_min_score_in_range():
    """retriever_domain_min_score는 [0, 1] 범위를 벗어나면 거부된다."""
    assert Settings(retriever_domain_min_score=0.0).retriever_domain_min_score == 0.0
    assert Settings(retriever_domain_min_score=1.0).retriever_domain_min_score == 1.0
    with pytest.raises(ValidationError):
        Settings(retriever_domain_min_score=1.5)
    with pytest.raises(ValidationError):
        Settings(retriever_domain_min_score=-0.1)


def test_domain_match_weight_non_negative():
    """retriever_domain_match_weight는 음수면 거부된다 (additive 가산이라 상한은 없음)."""
    assert Settings(retriever_domain_match_weight=2.0).retriever_domain_match_weight == 2.0
    with pytest.raises(ValidationError):
        Settings(retriever_domain_match_weight=-0.1)


def test_reranker_backend_literal_and_onnx_requires_dir():
    """reranker_backend는 torch/onnx만 허용하고, onnx면 onnx_dir이 필수다 (fail-fast)."""
    assert Settings(reranker_backend="torch").reranker_backend == "torch"
    assert Settings(reranker_backend="onnx", reranker_onnx_dir="models/x").reranker_backend == "onnx"
    with pytest.raises(ValidationError):
        Settings(reranker_backend="onnx")  # onnx_dir 없음 → fail-fast
    with pytest.raises(ValidationError):
        Settings(reranker_backend="onnnx")  # 오타 → Literal 거부
