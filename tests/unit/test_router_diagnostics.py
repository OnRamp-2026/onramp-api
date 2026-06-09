"""라우터 진단 순수 계산 테스트 (LLM/IO 불필요)."""

from app.eval import router_diagnostics as rd

ROWS = [
    {
        "is_answerable": True,
        "gold_domain": "incident",
        "pred_domain": "manual",
        "confidence": 0.9,
        "parse_ok": True,
        "failure_type": None,
    },
    {
        "is_answerable": True,
        "gold_domain": "incident",
        "pred_domain": "incident",
        "confidence": 0.8,
        "parse_ok": True,
        "failure_type": None,
    },
    {
        "is_answerable": True,
        "gold_domain": "manual",
        "pred_domain": "manual",
        "confidence": 0.6,
        "parse_ok": True,
        "failure_type": None,
    },
    {
        "is_answerable": True,
        "gold_domain": "manual",
        "pred_domain": "api_reference",
        "confidence": 0.4,
        "parse_ok": True,
        "failure_type": None,
    },
    {
        "is_answerable": True,
        "gold_domain": "api_reference",
        "pred_domain": None,
        "confidence": 0.0,
        "parse_ok": False,
        "failure_type": "json_decode",
    },
    {
        "is_answerable": False,
        "gold_domain": None,
        "pred_domain": None,
        "confidence": 0.0,
        "parse_ok": False,
        "failure_type": "llm_call",
    },
]


def test_scored_rows_excludes_unanswerable_and_nondomain():
    assert len(rd.scored_rows(ROWS)) == 5  # row5(gold None/unanswerable) 제외


def test_overall_accuracy():
    assert rd.overall_accuracy(ROWS) == (2, 5)  # incident·manual 정답 2건 / 5


def test_confusion_matrix():
    m = rd.confusion_matrix(ROWS)
    assert m["incident"]["incident"] == 1 and m["incident"]["manual"] == 1
    assert m["manual"]["manual"] == 1 and m["manual"]["api_reference"] == 1
    assert m["api_reference"][None] == 1
    assert sum(m["meeting_note"].values()) == 0


def test_per_domain_prf_macro_supported_vs_all():
    per, macro_sup, macro_all = rd.per_domain_prf(rd.confusion_matrix(ROWS))
    assert per["incident"]["recall"] == 0.5 and abs(per["incident"]["f1"] - 2 / 3) < 1e-9
    assert per["manual"]["precision"] == 0.5
    assert per["api_reference"]["f1"] == 0.0
    # 지원 3도메인 평균 vs 전체 5도메인 평균(0 표본 2개 포함)
    assert abs(macro_sup - (2 / 3 + 0.5 + 0.0) / 3) < 1e-9
    assert abs(macro_all - (2 / 3 + 0.5 + 0.0) / 5) < 1e-9
    assert macro_sup > macro_all  # 미커버 도메인이 전체 macro를 끌어내림


def test_calibration_and_ece():
    table, ece = rd.calibration(ROWS)
    by_bin = {t["bin"]: t for t in table}
    assert by_bin["0.9–1.0"]["accuracy"] == 0.0  # conf 0.9인데 오답
    assert by_bin["0.7–0.9"]["accuracy"] == 1.0
    assert abs(ece - 0.38) < 1e-9


def test_confusion_pairs_and_failures():
    pairs = dict(rd.confusion_pairs(ROWS))
    assert pairs[("incident", "manual")] == 1
    assert pairs[("manual", "api_reference")] == 1
    assert dict(rd.failure_summary(ROWS)) == {"json_decode": 1, "llm_call": 1}
