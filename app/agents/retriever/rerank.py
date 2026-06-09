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


def apply_metadata_weight(rerank_score: float, payload: dict, settings: Settings) -> float:
    """최신성 가산 — 최근 문서일수록 점수를 더한다. 가산식이라 음수 점수에서도 단조 증가한다."""
    factor = _recency_factor(payload.get("last_modified", ""), settings.rerank_recency_half_life_days)
    return rerank_score + settings.rerank_recency_weight * factor


def payload_domains(payload: dict) -> list[str]:
    """문서가 걸친 도메인 집합. 멀티도메인 `domains[]` 우선, 없으면 단일 `domain`(하위호환)."""
    domains = payload.get("domains")
    if domains:
        return list(domains)
    single = payload.get("domain")
    return [single] if single else []


def apply_domain_weight(rerank_score: float, payload: dict, domain: str | None, settings: Settings) -> float:
    """라우터 도메인이 문서 도메인 집합에 들면 점수를 더한다(Soft 가산).

    가산식이라 음수 점수(Cross-Encoder logit)에서도 단조 증가. domain이 None이거나
    문서 domains에 없으면 원점수를 그대로 반환한다. 멀티도메인 `domains[]`/단일 `domain` 모두 지원.
    """
    if domain and domain in payload_domains(payload):
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


_reranker: CrossEncoderReranker | None = None


def get_reranker(settings: Settings | None = None) -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(settings)
    return _reranker


def reset_reranker() -> None:
    global _reranker
    _reranker = None
