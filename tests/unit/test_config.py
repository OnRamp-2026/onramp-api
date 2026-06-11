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


def test_domain_weights_non_negative_and_primary_gt_secondary():
    """domain primary/secondary 가중치는 음수면 거부, 기본은 primary > secondary."""
    assert Settings(domain_primary_weight=2.0).domain_primary_weight == 2.0
    assert Settings(domain_secondary_weight=0.0).domain_secondary_weight == 0.0
    with pytest.raises(ValidationError):
        Settings(domain_primary_weight=-0.1)
    with pytest.raises(ValidationError):
        Settings(domain_secondary_weight=-0.1)
    s = Settings()
    assert s.domain_primary_weight > s.domain_secondary_weight


def test_reranker_backend_literal_and_onnx_requires_dir(tmp_path):
    """reranker_backend는 torch/onnx만 허용하고, onnx면 실제 모델 파일이 존재해야 한다 (fail-fast)."""
    assert Settings(reranker_backend="torch").reranker_backend == "torch"

    # 실제 모델 파일이 있으면 통과
    model_file = tmp_path / "model_quantized.onnx"
    model_file.write_bytes(b"")
    ok = Settings(reranker_backend="onnx", reranker_onnx_dir=str(tmp_path), reranker_onnx_file=model_file.name)
    assert ok.reranker_backend == "onnx"

    with pytest.raises(ValidationError):
        Settings(reranker_backend="onnx")  # onnx_dir 없음 → fail-fast
    with pytest.raises(ValidationError):
        Settings(reranker_backend="onnx", reranker_onnx_dir=str(tmp_path), reranker_onnx_file="  ")  # 파일 공백 → 거부
    with pytest.raises(ValidationError):
        Settings(
            reranker_backend="onnx", reranker_onnx_dir=str(tmp_path), reranker_onnx_file="absent.onnx"
        )  # 파일 미존재 → 거부
    with pytest.raises(ValidationError):
        Settings(reranker_backend="onnnx")  # 오타 → Literal 거부
