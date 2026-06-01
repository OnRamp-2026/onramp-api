"""bge-reranker Cross-Encoder 리랭킹 + 메타 가중 (검색측)."""

from __future__ import annotations

from datetime import UTC, datetime

from app.config import Settings, get_settings


class CrossEncoderReranker:
    """bge-reranker-v2-m3 Cross-Encoder. 모델은 첫 rerank 시 lazy-load."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = None

    @property
    def model(self):
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


def apply_metadata_weight(rerank_score: float, payload: dict, settings: Settings) -> float:
    """최신성 가중 — 최근 문서일수록 소폭 가산 (상한 rerank_recency_weight, rerank 순서 우선)."""
    factor = _recency_factor(payload.get("last_modified", ""), settings.rerank_recency_half_life_days)
    return rerank_score * (1 + settings.rerank_recency_weight * factor)


def _recency_factor(last_modified: str, half_life_days: int) -> float:
    try:
        dt = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    age_days = max((datetime.now(UTC) - dt).days, 0)
    return float(0.5 ** (age_days / half_life_days))


_reranker: CrossEncoderReranker | None = None


def get_reranker(settings: Settings | None = None) -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(settings)
    return _reranker


def reset_reranker() -> None:
    global _reranker
    _reranker = None
