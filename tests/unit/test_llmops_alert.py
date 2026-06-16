"""scripts/llmops_alert.py — Metrics API 임계 평가 (I3)."""

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path


def _load_mod():
    path = Path(__file__).resolve().parents[2] / "scripts" / "llmops_alert.py"
    spec = importlib.util.spec_from_file_location("llmops_alert", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["llmops_alert"] = mod  # dataclass(PEP563) 주석 해석을 위해 등록
    spec.loader.exec_module(mod)
    return mod


def test_thresholds_from_env(monkeypatch):
    mod = _load_mod()
    monkeypatch.setenv("ALERT_COST_1H_USD", "9.5")
    monkeypatch.setenv("ALERT_TRUST_MIN", "0.7")
    th = mod.Thresholds.from_env()
    assert th.cost_1h_usd == 9.5
    assert th.trust_min == 0.7


def test_evaluate_flags_cost_and_trust(monkeypatch):
    mod = _load_mod()

    def fake_query(client, query):
        if query["view"] == "observations":
            return [{"sum_totalCost": 9.0}]
        if query["view"] == "scores-numeric":
            return [{"name": "trust_score", "avg_value": 0.4}, {"name": "user_feedback", "avg_value": 1}]
        return []

    monkeypatch.setattr(mod, "query_metric", fake_query)
    breaches = mod.evaluate(client=None, now=datetime(2026, 6, 16, tzinfo=UTC), th=mod.Thresholds())
    assert len(breaches) == 2
    assert any("비용" in b for b in breaches)
    assert any("trust_score" in b for b in breaches)


def test_evaluate_no_breach_when_healthy(monkeypatch):
    mod = _load_mod()

    def fake_query(client, query):
        if query["view"] == "observations":
            return [{"sum_totalCost": 0.5}]
        if query["view"] == "scores":
            return [{"avg_value": 0.8}]
        return []

    monkeypatch.setattr(mod, "query_metric", fake_query)
    breaches = mod.evaluate(client=None, now=datetime(2026, 6, 16, tzinfo=UTC), th=mod.Thresholds())
    assert breaches == []


def test_notify_logs_without_webhook(capsys):
    mod = _load_mod()
    mod.notify(None, ["테스트 위반"])
    out = capsys.readouterr().out
    assert "로그만" in out and "테스트 위반" in out
