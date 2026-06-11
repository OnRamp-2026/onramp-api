"""멀티도메인 라우터 지표 단위 테스트 (순수 함수 + 분모 0 처리)."""

from app.eval.router_metrics import (
    RouterPred,
    expected_calibration_error,
    secondary_over_prediction_rate,
    secondary_precision,
    secondary_under_prediction_rate,
    summarize,
)

# 결정적 표본 4건
S = [
    RouterPred("a", ("incident", "manual"), ("incident", "manual"), 0.9, True),  # primary·ordered·set 정답
    RouterPred("b", ("incident",), ("manual",), 0.8, True),  # primary 오답
    RouterPred("c", ("incident", "manual"), ("incident",), 0.6, True),  # primary 정답·under-predict
    RouterPred("d", ("api_reference",), (), None, False),  # pred 빈값·parse 실패
]


def test_empty_samples_safe():
    m = summarize([])
    assert m.n_eval == 0
    assert m.primary_accuracy is None
    assert m.exact_ordered is None and m.exact_set is None
    assert m.micro == (0.0, 0.0, 0.0) and m.macro_label == (0.0, 0.0, 0.0)
    assert m.secondary_precision is None
    assert m.secondary_over_rate is None and m.secondary_under_rate is None
    assert m.ece == 0.0 and m.ece_n_used == 0
    m.as_dict()  # 직렬화 예외 없음


def test_primary_and_exact_matches():
    m = summarize(S)
    assert m.n_eval == 4
    assert m.primary_accuracy == 0.5  # a, c 정답 / 4
    assert m.exact_ordered == 0.25  # a 만
    assert m.exact_set == 0.25  # a 만


def test_secondary_metrics():
    assert secondary_precision(S) == 1.0  # 예측 secondary는 a의 manual 하나 → gold에 있음
    assert secondary_over_prediction_rate(S) == 0.0  # secondary 예측 질문 a 1건, spurious 아님
    assert secondary_under_prediction_rate(S) == 0.5  # gold secondary 보유 a,c 중 c가 놓침


def test_secondary_none_when_no_denominator():
    single = [RouterPred("x", ("incident",), ("incident",), 0.9, True)]
    assert secondary_precision(single) is None  # 예측 secondary 없음
    assert secondary_over_prediction_rate(single) is None
    assert secondary_under_prediction_rate(single) is None  # gold secondary 없음


def test_ece_excludes_failed_and_none_confidence():
    ece, n_used, n_excluded = expected_calibration_error(S)
    assert n_used == 3  # a,b,c (parse_ok ∧ confidence 보유)
    assert n_excluded == 1  # d (parse 실패·conf None)
    assert 0.0 <= ece <= 1.0


def test_raw_vs_effective_low_confidence_distinction():
    # 저신뢰이지만 분류는 맞은 경우: raw(게이팅 전)는 정답, effective(게이팅 후)는 빈 예측=오답.
    # 둘을 분리해야 "분류가 틀린 것"과 "도메인은 맞지만 저신뢰로 비워진 것"을 구분할 수 있다.
    gold = ("incident",)
    raw = [RouterPred("x", gold, ("incident",), 0.4, True)]  # 라우터 분류 능력: 정답
    eff = [RouterPred("x", gold, (), 0.4, True, low_conf_empty=True)]  # 게이팅 후 운영 결과: 빔
    assert summarize(raw).primary_accuracy == 1.0
    assert summarize(eff).primary_accuracy == 0.0
    # calibration(ECE)은 raw 기준만 유효 — effective는 게이팅으로 빈 예측을 오답 처리해 왜곡된다.
    assert expected_calibration_error(raw)[0] != expected_calibration_error(eff)[0]


def test_summary_counts_passthrough():
    m = summarize(S, parse_failures=1, low_conf_empty=2, extras={"unanswerable_block_accuracy": 0.5})
    d = m.as_dict()
    assert d["parse_failures"] == 1 and d["low_conf_empty"] == 2
    assert d["unanswerable_block_accuracy"] == 0.5
    assert d["ece_n_excluded"] == 1
