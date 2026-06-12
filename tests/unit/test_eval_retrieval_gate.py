"""검색 회귀 게이트 _check_gate 단위 테스트 (n-불일치 감지 포함)."""

import json

from scripts.eval_retrieval import GATED_MODE, _check_gate


def _baseline(tmp_path, n, metrics):
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps({GATED_MODE: metrics, "n": n}), encoding="utf-8")
    return p


def test_gate_fails_on_n_mismatch(tmp_path):
    # 골든셋이 바뀌어 평가 문항 수가 다르면(82≠81) 지표가 같아도 게이트 실패
    base = _baseline(tmp_path, 82, {"recall@5": 0.5})
    report = {GATED_MODE: {"recall@5": 0.5}, "n": 81}
    assert _check_gate(report, base, tolerance=0.01) == 1


def test_gate_passes_when_n_matches_and_no_regression(tmp_path):
    base = _baseline(tmp_path, 81, {"recall@5": 0.5})
    report = {GATED_MODE: {"recall@5": 0.5}, "n": 81}
    assert _check_gate(report, base, tolerance=0.01) == 0


def test_gate_fails_on_regression(tmp_path):
    base = _baseline(tmp_path, 81, {"recall@5": 0.5})
    report = {GATED_MODE: {"recall@5": 0.3}, "n": 81}  # recall 회귀
    assert _check_gate(report, base, tolerance=0.01) == 1
