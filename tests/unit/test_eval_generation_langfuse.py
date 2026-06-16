"""scripts/eval_generation_langfuse.py — RAGAS run_evaluator 로직 (#120)."""

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_mod():
    path = Path(__file__).resolve().parents[2] / "scripts" / "eval_generation_langfuse.py"
    spec = importlib.util.spec_from_file_location("eval_generation_langfuse", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_generation_langfuse"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_run_requires_ragas(monkeypatch):
    mod = _load_mod()
    monkeypatch.setattr(mod, "ragas_available", lambda: False)
    assert mod.run() == 1


def test_run_requires_langfuse(monkeypatch):
    mod = _load_mod()
    monkeypatch.setattr(mod, "ragas_available", lambda: True)
    monkeypatch.setattr(mod, "get_langfuse_client", lambda: None)
    assert mod.run() == 1


@pytest.mark.asyncio
async def test_run_evaluator_builds_faithfulness_and_relevancy(monkeypatch):
    mod = _load_mod()

    class _Scores:
        faithfulness = 0.9
        answer_relevancy = 0.8

    async def fake_score(results, **kw):
        assert len(results) == 1  # evaluable 1건만
        return _Scores()

    monkeypatch.setattr(mod, "score_generation", fake_score)

    class _IR:
        def __init__(self, o):
            self.output = o

    item_results = [
        _IR({"evaluable": True, "query": "q", "answer_text": "a", "retrieved_contexts": ["c"], "reference": None}),
        _IR({"evaluable": False}),  # 제외
    ]
    evals = await mod._ragas_run_evaluator(item_results=item_results)
    assert {e.name for e in evals} == {"faithfulness", "answer_relevancy"}
    assert {e.value for e in evals} == {0.9, 0.8}


@pytest.mark.asyncio
async def test_run_evaluator_empty_when_none_evaluable(monkeypatch):
    mod = _load_mod()

    class _IR:
        output = {"evaluable": False}

    assert await mod._ragas_run_evaluator(item_results=[_IR()]) == []
