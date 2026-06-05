"""검색 평가 결정론 지표 (순수 함수, I/O·LLM 없음).

관련성 단위는 chunk_id. 각 함수는 순위가 매겨진 chunk_id 리스트(`ranked`)와
정답 chunk_id 집합(`relevant`)을 받아 0~1 점수를 반환한다.
unanswerable(=relevant 빈셋) 질문은 검색 지표 집계에서 제외하고,
answerability 정확도로 따로 평가한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _validate_k(k: int, name: str = "k") -> None:
    """k 는 1 이상이어야 한다 (음수 슬라이싱으로 인한 조용한 왜곡 방지)."""
    if k <= 0:
        raise ValueError(f"{name} 는 1 이상이어야 합니다: {k}")


def hit_rate_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """top-k 안에 정답 chunk가 하나라도 있으면 1.0, 없으면 0.0."""
    _validate_k(k)
    if not relevant:
        return 0.0
    return 1.0 if any(c in relevant for c in ranked[:k]) else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str], k: int) -> float:
    """top-k 내 첫 정답의 역순위(1/rank). 없으면 0.0."""
    _validate_k(k)
    if not relevant:
        return 0.0
    for i, chunk_id in enumerate(ranked[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """top-k 가 정답 집합을 얼마나 덮는가 (|정답 ∩ top-k| / |정답|)."""
    _validate_k(k)
    if not relevant:
        return 0.0
    found = sum(1 for c in set(ranked[:k]) if c in relevant)
    return found / len(relevant)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """binary-gain nDCG@k. 없으면 0.0."""
    _validate_k(k)
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, chunk_id in enumerate(ranked[:k], start=1):
        if chunk_id in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass(frozen=True)
class MetricSummary:
    """매크로 평균 검색 지표 요약 (relevant 빈셋 질문은 n에서 제외)."""

    n: int
    hit_rate: float
    mrr: float
    recall: float
    ndcg: float
    k_hit: int
    k_mrr: int
    k_recall: int
    k_ndcg: int

    def as_dict(self) -> dict[str, float]:
        return {
            f"hit_rate@{self.k_hit}": round(self.hit_rate, 4),
            f"mrr@{self.k_mrr}": round(self.mrr, 4),
            f"recall@{self.k_recall}": round(self.recall, 4),
            f"ndcg@{self.k_ndcg}": round(self.ndcg, 4),
        }


def aggregate(
    per_query: list[tuple[list[str], set[str]]],
    *,
    k_hit: int = 5,
    k_mrr: int = 10,
    k_recall: int = 5,
    k_ndcg: int = 10,
) -> MetricSummary:
    """질문별 (ranked, relevant) 목록을 매크로 평균한다.

    relevant 가 빈셋(unanswerable)인 질문은 검색 지표 대상에서 제외한다.
    """
    _validate_k(k_hit, "k_hit")
    _validate_k(k_mrr, "k_mrr")
    _validate_k(k_recall, "k_recall")
    _validate_k(k_ndcg, "k_ndcg")
    scored = [(ranked, rel) for ranked, rel in per_query if rel]
    n = len(scored)
    if n == 0:
        return MetricSummary(0, 0.0, 0.0, 0.0, 0.0, k_hit, k_mrr, k_recall, k_ndcg)

    hit = sum(hit_rate_at_k(r, rel, k_hit) for r, rel in scored) / n
    mrr = sum(reciprocal_rank(r, rel, k_mrr) for r, rel in scored) / n
    rec = sum(recall_at_k(r, rel, k_recall) for r, rel in scored) / n
    ndcg = sum(ndcg_at_k(r, rel, k_ndcg) for r, rel in scored) / n
    return MetricSummary(n, hit, mrr, rec, ndcg, k_hit, k_mrr, k_recall, k_ndcg)


@dataclass(frozen=True)
class AnswerabilitySummary:
    """답변가능 판정 정확도 (positive = answerable)."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def answerability_accuracy(preds: list[bool], labels: list[bool]) -> AnswerabilitySummary:
    """예측 answerable(preds) 대 정답 is_answerable(labels) 혼동행렬·지표."""
    if len(preds) != len(labels):
        raise ValueError("preds 와 labels 길이가 다릅니다")
    tp = sum(1 for p, t in zip(preds, labels, strict=True) if p and t)
    fp = sum(1 for p, t in zip(preds, labels, strict=True) if p and not t)
    tn = sum(1 for p, t in zip(preds, labels, strict=True) if not p and not t)
    fn = sum(1 for p, t in zip(preds, labels, strict=True) if not p and t)
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return AnswerabilitySummary(accuracy, precision, recall, f1, tp, fp, tn, fn)
