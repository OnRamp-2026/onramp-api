"""bge-reranker Cross-Encoder 리랭킹 + 메타 가중 (검색측)."""

from __future__ import annotations

import threading
from datetime import UTC, datetime

from app.config import Settings, get_settings


class CrossEncoderReranker:
    """bge-reranker-v2-m3 Cross-Encoder. 모델은 첫 rerank 시 lazy-load."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        # double-checked locking — anyio 스레드에서 동시 cold-start 시 모델 중복 로드 방지
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder  # 무거운 로드 → 지연

                    self._model = CrossEncoder(self.settings.reranker_model, device=self.settings.reranker_device)
        return self._model

    def rerank(self, query: str, candidates: list[tuple[str, dict]]) -> list[tuple[float, dict]]:
        if not candidates:
            return []
        pairs = [(query, text) for text, _ in candidates]
        scores = self.model.predict(pairs)
        ranked = [(float(score), payload) for score, (_, payload) in zip(scores, candidates, strict=True)]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked


class OnnxCrossEncoderReranker:
    """#60: ONNX(int8) bge-reranker. CrossEncoderReranker와 동일 인터페이스, CPU 파드 경량화용.

    검증 범위(standalone): 변환(fp32→int8) + CPU 속도 + 골든셋 품질만 확인.
    동일 모델 그대로 양자화하므로 다국어 보존. in-app 경로는 운영 파드에서 재검증 필요(그래서 기본 backend는 torch).
    모델 디렉토리는 scripts/build_reranker_onnx.py로 사전 생성한다(산출물 model_quantized.onnx).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is None:
            with self._lock:  # double-checked locking — 동시 cold-start 시 중복 로드 방지
                if self._model is None:
                    if not self.settings.reranker_onnx_dir:
                        raise RuntimeError(
                            "reranker_backend='onnx'인데 reranker_onnx_dir 미설정 "
                            "(scripts/build_reranker_onnx.py 산출물 경로 지정 필요)"
                        )
                    from optimum.onnxruntime import ORTModelForSequenceClassification  # 무거운 로드 → 지연
                    from transformers import AutoTokenizer

                    self._tokenizer = AutoTokenizer.from_pretrained(self.settings.reranker_model)
                    self._model = ORTModelForSequenceClassification.from_pretrained(
                        self.settings.reranker_onnx_dir,
                        file_name=self.settings.reranker_onnx_file,
                        provider="CPUExecutionProvider",
                    )

    def rerank(self, query: str, candidates: list[tuple[str, dict]]) -> list[tuple[float, dict]]:
        if not candidates:
            return []
        self._ensure_loaded()
        import torch  # 지연

        passages = [text for text, _ in candidates]
        features = self._tokenizer(  # type: ignore[misc]
            [query] * len(passages),
            passages,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**features).logits  # type: ignore[misc]
        scores = logits.squeeze(-1).cpu().tolist()
        ranked = [(float(score), payload) for score, (_, payload) in zip(scores, candidates, strict=True)]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked


def apply_metadata_weight(rerank_score: float, payload: dict, settings: Settings) -> float:
    """최신성 가산 — 최근 문서일수록 점수를 더한다. 가산식이라 음수 점수에서도 단조 증가한다."""
    factor = _recency_factor(payload.get("last_modified", ""), settings.rerank_recency_half_life_days)
    return rerank_score + settings.rerank_recency_weight * factor


def apply_domain_weight(rerank_score: float, payload: dict, domain: str | None, settings: Settings) -> float:
    """도메인이 일치하면 점수를 더한다. 가산식이라 음수 점수(Cross-Encoder logit)에서도 단조 증가한다.

    domain이 None이거나 불일치면 원점수를 그대로 반환한다.
    """
    if domain and payload.get("domain") == domain:
        return rerank_score + settings.retriever_domain_match_weight
    return rerank_score


def _recency_factor(last_modified: str, half_life_days: int) -> float:
    try:
        dt = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    age_days = max((datetime.now(UTC) - dt).days, 0)
    return float(0.5 ** (age_days / half_life_days))


# Trust Agent(Evidence Confidence)도 동일한 최신성 계수를 쓴다 — 공개 별칭.
recency_factor = _recency_factor


_reranker: CrossEncoderReranker | OnnxCrossEncoderReranker | None = None


def get_reranker(settings: Settings | None = None) -> CrossEncoderReranker | OnnxCrossEncoderReranker:
    global _reranker
    if _reranker is None:
        cfg = settings or get_settings()
        if cfg.reranker_backend == "onnx":  # #60: int8 경량화 백엔드(opt-in)
            _reranker = OnnxCrossEncoderReranker(settings)
        else:
            _reranker = CrossEncoderReranker(settings)
    return _reranker


def reset_reranker() -> None:
    global _reranker
    _reranker = None
