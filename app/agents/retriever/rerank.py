"""bge-reranker Cross-Encoder 리랭킹 + 메타 가중 (검색측)."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
        self._session: Any = None  # onnxruntime.InferenceSession (지연 로드)
        self._tokenizer: Any = None  # transformers tokenizer (지연 로드)
        self._input_names: set[str] = set()
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        # 순수 onnxruntime + numpy로 추론 — torch 의존 없음([onnx] extra만으로 동작).
        if self._session is None:
            with self._lock:  # double-checked locking — 동시 cold-start 시 중복 로드 방지
                if self._session is None:
                    model_path = Path(self.settings.reranker_onnx_dir, self.settings.reranker_onnx_file)
                    if not model_path.is_file():
                        # config 검증을 우회한 경우의 2차 방어선(테스트/직접 생성). 정상 기동은 config가 먼저 막는다.
                        raise RuntimeError(
                            f"reranker_backend='onnx' 모델 파일 없음: {model_path} "
                            "(scripts/build_reranker_onnx.py 먼저 실행)"
                        )
                    import onnxruntime as ort  # 무거운 로드 → 지연
                    from transformers import AutoTokenizer

                    self._tokenizer = AutoTokenizer.from_pretrained(self.settings.reranker_model)
                    self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                    self._input_names = {i.name for i in self._session.get_inputs()}

    def rerank(self, query: str, candidates: list[tuple[str, dict]]) -> list[tuple[float, dict]]:
        if not candidates:
            return []
        self._ensure_loaded()
        import numpy as np  # 지연

        passages = [text for text, _ in candidates]
        features = self._tokenizer(
            [query] * len(passages),
            passages,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        # 양자화 그래프가 받는 입력만 전달 (모델별 token_type_ids 유무 차이 흡수)
        inputs = {k: v for k, v in features.items() if k in self._input_names}
        logits = self._session.run(None, inputs)[0]
        scores = (1.0 / (1.0 + np.exp(-logits))).reshape(-1).tolist()  # sigmoid: torch 백엔드와 동일 점수 계약
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
_reranker_key: tuple[str, str, str, str, str] | None = None
_reranker_lock = threading.Lock()  # _reranker/_reranker_key 동시 갱신 보호 (key↔instance 불일치 방지)


def get_reranker(settings: Settings | None = None) -> CrossEncoderReranker | OnnxCrossEncoderReranker:
    # backend/model/device/artifact 조합이 바뀌면 재생성 (torch↔onnx·CPU↔GPU 전환·테스트 격리 보장)
    global _reranker, _reranker_key
    cfg = settings or get_settings()
    key = (cfg.reranker_backend, cfg.reranker_model, cfg.reranker_device, cfg.reranker_onnx_dir, cfg.reranker_onnx_file)
    with _reranker_lock:  # 동시 초기화·설정 전환에서 key와 instance가 어긋난 채 반환되는 것을 막는다
        if _reranker is None or _reranker_key != key:
            _reranker_key = key
            if cfg.reranker_backend == "onnx":  # #60: int8 경량화 백엔드(opt-in)
                _reranker = OnnxCrossEncoderReranker(cfg)
            else:
                _reranker = CrossEncoderReranker(cfg)
        return _reranker


def reset_reranker() -> None:
    global _reranker, _reranker_key
    with _reranker_lock:
        _reranker = None
        _reranker_key = None
