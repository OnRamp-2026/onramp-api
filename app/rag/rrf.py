"""Reciprocal Rank Fusion helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RankedItem:
    id: str
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class FusedItem:
    id: str
    score: float
    payload: dict[str, Any]
    source_scores: dict[str, float]


def reciprocal_rank_fusion(
    ranked_lists: Iterable[tuple[str, list[RankedItem]]],
    *,
    k: int = 60,
    limit: int | None = None,
) -> list[FusedItem]:
    """Merge ranked result lists using RRF.

    The original provider score is retained as source_scores, while the final
    score is rank-based and therefore comparable across dense/BM25 providers.
    """

    fused: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}
    source_scores: dict[str, dict[str, float]] = {}

    for source, items in ranked_lists:
        for rank, item in enumerate(items, start=1):
            fused[item.id] = fused.get(item.id, 0.0) + 1.0 / (k + rank)
            payloads.setdefault(item.id, item.payload)
            source_scores.setdefault(item.id, {})[source] = item.score

    ordered = [
        FusedItem(id=item_id, score=score, payload=payloads[item_id], source_scores=source_scores[item_id])
        for item_id, score in fused.items()
    ]
    ordered.sort(key=lambda item: item.score, reverse=True)
    return ordered[:limit] if limit is not None else ordered
