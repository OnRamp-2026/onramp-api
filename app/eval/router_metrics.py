"""멀티도메인 라우터 평가 지표 (순수 함수, I/O·LLM 없음).

비교 대상:
    gold = `GoldenQuery.router_domains` (사람 검수 정답, 순서=우선순위)
    pred = 라우터 예측 캐시의 `predicted_domains` (confidence 게이팅 후)

평가집합 = **answerable ∧ router_domains 비어있지 않은** 질문. pred가 비어도(저신뢰/실패)
표본에는 포함하되 primary는 오답으로 센다 — 단, parse failure·low-confidence-empty 수는
별도로 집계한다. 모든 분모 0은 예외 없이 0.0 또는 None(N/A)으로 명확히 반환한다.

macro 명명(혼동 방지):
    · micro_*       = 전체 (질문, 라벨) 쌍을 합산한 TP/FP/FN 기반
    · macro_label_* = **도메인(라벨)별** P/R/F1의 평균 (label-macro)
    (질문별 P/R/F1을 평균하는 sample-macro와 다르다 — 이번엔 micro·label-macro만 낸다.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.agents.state import Domain

ALL_DOMAINS: tuple[str, ...] = tuple(d.value for d in Domain)

# confidence 구간 (하한 포함, 상한 미포함, 마지막 구간만 1.0 포함)
_CONF_BINS: tuple[tuple[float, float], ...] = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01))


@dataclass(frozen=True)
class RouterPred:
    """평가 표본 한 건 (예측 캐시 + 골든 정답 조인 결과)."""

    qid: str
    gold: tuple[str, ...]  # router_domains (정답, 비어있지 않음)
    pred: tuple[str, ...]  # predicted_domains (게이팅 후, 비어 있을 수 있음)
    confidence: float | None  # 실패면 None
    parse_ok: bool
    low_conf_empty: bool = False  # parse_ok인데 저신뢰로 pred가 비워졌는가


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """precision/recall/f1. 분모 0이면 0.0."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def primary_accuracy(samples: list[RouterPred]) -> tuple[int, int]:
    """pred[0] == gold[0] 인 표본 수와 분모(전체 표본). pred 비면 오답."""
    correct = sum(1 for s in samples if s.pred and s.pred[0] == s.gold[0])
    return correct, len(samples)


def exact_ordered_match(samples: list[RouterPred]) -> tuple[int, int]:
    """pred == gold (순서 포함) 인 표본 수와 분모."""
    correct = sum(1 for s in samples if tuple(s.pred) == tuple(s.gold))
    return correct, len(samples)


def exact_set_match(samples: list[RouterPred]) -> tuple[int, int]:
    """set(pred) == set(gold) (순서 무시) 인 표본 수와 분모."""
    correct = sum(1 for s in samples if set(s.pred) == set(s.gold))
    return correct, len(samples)


def micro_prf(samples: list[RouterPred]) -> tuple[float, float, float]:
    """전체 (질문, 라벨) 쌍 합산 micro P/R/F1."""
    tp = fp = fn = 0
    for s in samples:
        gold, pred = set(s.gold), set(s.pred)
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    return _prf(tp, fp, fn)


def per_domain_prf(samples: list[RouterPred], domains: tuple[str, ...] = ALL_DOMAINS) -> dict[str, dict[str, float]]:
    """도메인별 P/R/F1 + support(정답에 등장한 횟수)."""
    out: dict[str, dict[str, float]] = {}
    for d in domains:
        tp = sum(1 for s in samples if d in s.pred and d in s.gold)
        fp = sum(1 for s in samples if d in s.pred and d not in s.gold)
        fn = sum(1 for s in samples if d not in s.pred and d in s.gold)
        p, r, f1 = _prf(tp, fp, fn)
        out[d] = {"precision": p, "recall": r, "f1": f1, "support": float(tp + fn)}
    return out


def macro_label_prf(samples: list[RouterPred], domains: tuple[str, ...] = ALL_DOMAINS) -> tuple[float, float, float]:
    """도메인(라벨)별 P/R/F1의 단순 평균 (label-macro)."""
    if not domains:
        return 0.0, 0.0, 0.0
    per = per_domain_prf(samples, domains)
    n = len(domains)
    p = sum(per[d]["precision"] for d in domains) / n
    r = sum(per[d]["recall"] for d in domains) / n
    f1 = sum(per[d]["f1"] for d in domains) / n
    return p, r, f1


def secondary_precision(samples: list[RouterPred]) -> float | None:
    """예측한 secondary(pred[1:]) 중 정답 집합에 든 비율. 예측 secondary가 없으면 None."""
    total = correct = 0
    for s in samples:
        for d in s.pred[1:]:
            total += 1
            correct += d in s.gold
    return correct / total if total else None


def secondary_over_prediction_rate(samples: list[RouterPred]) -> float | None:
    """secondary를 예측한 질문 중, 그 secondary가 정답에 없는 비율. 예측 질문이 없으면 None."""
    denom = [s for s in samples if len(s.pred) >= 2]
    if not denom:
        return None
    spurious = sum(1 for s in denom if s.pred[1] not in s.gold)
    return spurious / len(denom)


def secondary_under_prediction_rate(samples: list[RouterPred]) -> float | None:
    """정답에 secondary가 있는 질문 중, 그 secondary를 예측 못 한 비율. 해당 질문이 없으면 None."""
    denom = [s for s in samples if len(s.gold) >= 2]
    if not denom:
        return None
    missed = sum(1 for s in denom if s.gold[1] not in s.pred)
    return missed / len(denom)


def _primary_correct(s: RouterPred) -> bool:
    return bool(s.pred) and s.pred[0] == s.gold[0]


def confidence_bins(samples: list[RouterPred]) -> list[dict[str, float | int | None]]:
    """confidence 구간별 (n, primary_accuracy, mean_confidence). parse_ok ∧ confidence 보유 표본만."""
    usable = [s for s in samples if s.parse_ok and s.confidence is not None]
    bins: list[dict[str, float | int | None]] = []
    for lo, hi in _CONF_BINS:
        members = [s for s in usable if s.confidence is not None and lo <= s.confidence < hi]
        n = len(members)
        acc = sum(_primary_correct(s) for s in members) / n if n else None
        mean_conf = sum(s.confidence for s in members if s.confidence is not None) / n if n else None
        bins.append({"lo": lo, "hi": min(hi, 1.0), "n": n, "primary_accuracy": acc, "mean_confidence": mean_conf})
    return bins


def expected_calibration_error(samples: list[RouterPred]) -> tuple[float, int, int]:
    """primary 정답 기준 ECE = Σ (n_b/N)·|acc_b − conf_b|.

    parse_ok ∧ confidence 보유 표본만 사용(실패·None은 ECE 왜곡 방지 위해 제외).
    반환: (ece, n_used, n_excluded).
    """
    usable = [s for s in samples if s.parse_ok and s.confidence is not None]
    excluded = len(samples) - len(usable)
    n = len(usable)
    if n == 0:
        return 0.0, 0, excluded
    ece = 0.0
    for lo, hi in _CONF_BINS:
        members = [s for s in usable if s.confidence is not None and lo <= s.confidence < hi]
        if not members:
            continue
        acc = sum(_primary_correct(s) for s in members) / len(members)
        conf = sum(s.confidence for s in members if s.confidence is not None) / len(members)
        ece += (len(members) / n) * abs(acc - conf)
    return ece, n, excluded


@dataclass(frozen=True)
class RouterMetrics:
    """라우터 멀티도메인 평가 요약 (answerable ∧ router_domains 보유 표본)."""

    n_eval: int
    primary_accuracy: float | None
    exact_ordered: float | None
    exact_set: float | None
    micro: tuple[float, float, float]
    macro_label: tuple[float, float, float]
    per_domain: dict[str, dict[str, float]]
    secondary_precision: float | None
    secondary_over_rate: float | None
    secondary_under_rate: float | None
    conf_bins: list[dict[str, float | int | None]]
    ece: float
    ece_n_used: int
    ece_n_excluded: int
    parse_failures: int = 0
    low_conf_empty: int = 0
    extras: dict[str, float | int | None] = field(default_factory=dict)

    def as_dict(self) -> dict:
        def rd(x: float | None) -> float | None:
            return round(x, 4) if x is not None else None

        p, r, f1 = self.micro
        mp, mr, mf1 = self.macro_label
        return {
            "n_eval": self.n_eval,
            "primary_accuracy": rd(self.primary_accuracy),
            "exact_ordered_match": rd(self.exact_ordered),
            "exact_set_match": rd(self.exact_set),
            "micro": {"precision": rd(p), "recall": rd(r), "f1": rd(f1)},
            "macro_label": {"precision": rd(mp), "recall": rd(mr), "f1": rd(mf1)},
            "per_domain": {d: {k: rd(v) for k, v in vals.items()} for d, vals in self.per_domain.items()},
            "secondary_precision": rd(self.secondary_precision),
            "secondary_over_prediction_rate": rd(self.secondary_over_rate),
            "secondary_under_prediction_rate": rd(self.secondary_under_rate),
            "confidence_bins": self.conf_bins,
            "ece": rd(self.ece),
            "ece_n_used": self.ece_n_used,
            "ece_n_excluded": self.ece_n_excluded,
            "parse_failures": self.parse_failures,
            "low_conf_empty": self.low_conf_empty,
            **self.extras,
        }


def summarize(
    samples: list[RouterPred],
    *,
    domains: tuple[str, ...] = ALL_DOMAINS,
    parse_failures: int = 0,
    low_conf_empty: int = 0,
    extras: dict[str, float | int | None] | None = None,
) -> RouterMetrics:
    """평가 표본을 종합해 RouterMetrics로 요약한다. 빈 표본도 안전(모두 None/0)."""
    n = len(samples)
    pa_c, pa_n = primary_accuracy(samples)
    eo_c, _ = exact_ordered_match(samples)
    es_c, _ = exact_set_match(samples)
    ece, ece_used, ece_excl = expected_calibration_error(samples)
    return RouterMetrics(
        n_eval=n,
        primary_accuracy=(pa_c / pa_n) if pa_n else None,
        exact_ordered=(eo_c / n) if n else None,
        exact_set=(es_c / n) if n else None,
        micro=micro_prf(samples),
        macro_label=macro_label_prf(samples, domains),
        per_domain=per_domain_prf(samples, domains),
        secondary_precision=secondary_precision(samples),
        secondary_over_rate=secondary_over_prediction_rate(samples),
        secondary_under_rate=secondary_under_prediction_rate(samples),
        conf_bins=confidence_bins(samples),
        ece=ece,
        ece_n_used=ece_used,
        ece_n_excluded=ece_excl,
        parse_failures=parse_failures,
        low_conf_empty=low_conf_empty,
        extras=extras or {},
    )
