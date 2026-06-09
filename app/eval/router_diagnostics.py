"""라우터 진단 — 순수 계산(테스트 가능). LLM/IO 없음.

입력 rows: list[dict] — 각 행에 최소
    gold_domain: str | None, pred_domain: str | None, confidence: float,
    is_answerable: bool, parse_ok: bool, failure_type: str | None
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

DOMAINS: tuple[str, ...] = ("incident", "manual", "api_reference", "meeting_note", "planning")
CONF_BINS: tuple[tuple[float, float, str], ...] = (
    (0.9, 1.01, "0.9–1.0"),
    (0.7, 0.9, "0.7–0.9"),
    (0.5, 0.7, "0.5–0.7"),
    (0.0, 0.5, "0.0–0.5"),
)


def scored_rows(rows: Sequence[dict], domains: Sequence[str] = DOMAINS) -> list[dict]:
    """검색 지표 대상 — answerable + gold 도메인이 5종 안에 있는 행만."""
    return [r for r in rows if r.get("is_answerable") and r.get("gold_domain") in domains]


def confusion_matrix(rows: Sequence[dict], domains: Sequence[str] = DOMAINS) -> dict[str, dict[str | None, int]]:
    """gold(행) × pred(열, None 포함) 카운트. scored 행만."""
    matrix: dict[str, dict[str | None, int]] = {g: dict.fromkeys([*domains, None], 0) for g in domains}
    for r in scored_rows(rows, domains):
        pred = r.get("pred_domain") if r.get("pred_domain") in domains else None
        matrix[r["gold_domain"]][pred] += 1
    return matrix


def per_domain_prf(
    matrix: dict[str, dict[str | None, int]], domains: Sequence[str] = DOMAINS
) -> tuple[dict[str, dict[str, float]], float, float]:
    """도메인별 P/R/F1 + macro-F1(지원도메인만), macro-F1(전체)을 반환한다.

    지원도메인 = gold 표본(support_gold)이 1건 이상인 도메인.
    """
    per: dict[str, dict[str, float]] = {}
    for d in domains:
        tp = matrix[d][d]
        support_gold = sum(matrix[d].values())
        support_pred = sum(matrix[g][d] for g in domains)
        recall = tp / support_gold if support_gold else 0.0
        precision = tp / support_pred if support_pred else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per[d] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support_gold": support_gold,
            "support_pred": support_pred,
        }
    supported = [d for d in domains if per[d]["support_gold"] > 0]
    macro_supported = sum(per[d]["f1"] for d in supported) / len(supported) if supported else 0.0
    macro_all = sum(per[d]["f1"] for d in domains) / len(domains) if domains else 0.0
    return per, macro_supported, macro_all


def calibration(
    rows: Sequence[dict], bins: Sequence[tuple[float, float, str]] = CONF_BINS, domains: Sequence[str] = DOMAINS
) -> tuple[list[dict], float]:
    """confidence bin별 (n, accuracy, mean_conf) + ECE(Expected Calibration Error).

    ECE = Σ (n_b/N) · |acc_b − conf_b|. scored 행 대상.
    """
    scored = scored_rows(rows, domains)
    total = len(scored)
    table: list[dict] = []
    ece = 0.0
    for lo, hi, label in bins:
        bucket = [r for r in scored if lo <= r.get("confidence", 0.0) < hi]
        n = len(bucket)
        if n == 0:
            table.append({"bin": label, "n": 0, "accuracy": 0.0, "mean_conf": 0.0})
            continue
        acc = sum(r["pred_domain"] == r["gold_domain"] for r in bucket) / n
        mean_conf = sum(r.get("confidence", 0.0) for r in bucket) / n
        table.append({"bin": label, "n": n, "accuracy": acc, "mean_conf": mean_conf})
        if total:
            ece += (n / total) * abs(acc - mean_conf)
    return table, ece


def overall_accuracy(rows: Sequence[dict], domains: Sequence[str] = DOMAINS) -> tuple[int, int]:
    """scored 행의 (정답수, 총수)."""
    scored = scored_rows(rows, domains)
    correct = sum(r["pred_domain"] == r["gold_domain"] for r in scored)
    return correct, len(scored)


def confusion_pairs(rows: Sequence[dict], domains: Sequence[str] = DOMAINS) -> list[tuple[tuple[str, str | None], int]]:
    """주요 오분류 방향 (gold→pred) 빈도 내림차순."""
    miss = Counter(
        (r["gold_domain"], r.get("pred_domain"))
        for r in scored_rows(rows, domains)
        if r.get("pred_domain") != r["gold_domain"]
    )
    return miss.most_common()


def failure_summary(rows: Sequence[dict]) -> Counter[str]:
    """파싱/호출 실패를 failure_type별로 집계."""
    return Counter(r.get("failure_type") or "unknown" for r in rows if not r.get("parse_ok", True))
