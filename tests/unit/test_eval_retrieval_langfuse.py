"""scripts/eval_retrieval_langfuse.py — 검색 experiment evaluator 로직 (#120)."""

import importlib.util
import sys
from pathlib import Path


def _load_mod():
    path = Path(__file__).resolve().parents[2] / "scripts" / "eval_retrieval_langfuse.py"
    spec = importlib.util.spec_from_file_location("eval_retrieval_langfuse", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_retrieval_langfuse"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_evaluator_scores_hit_and_rank():
    mod = _load_mod()
    ev_hit = mod._make_evaluator("hit_rate@5", mod.metrics.hit_rate_at_k)
    # 정답 c2가 top-5 안 → hit 1.0
    out = ev_hit(input={}, output=["c1", "c2", "c3"], expected_output={"relevant_chunk_ids": ["c2"]})
    assert out.name == "hit_rate@5"
    assert out.value == 1.0
    # 정답 없음(top-5 밖) → 0.0
    miss = ev_hit(input={}, output=["x1", "x2"], expected_output={"relevant_chunk_ids": ["zzz"]})
    assert miss.value == 0.0


def test_evaluator_skips_unanswerable():
    mod = _load_mod()
    ev = mod._make_evaluator("mrr@5", mod.metrics.reciprocal_rank)
    # relevant 빈셋(unanswerable) → 평가 제외(빈 리스트)
    assert ev(input={}, output=["a"], expected_output={"relevant_chunk_ids": []}) == []
    assert ev(input={}, output=["a"], expected_output={}) == []


def test_run_requires_langfuse(monkeypatch):
    mod = _load_mod()
    monkeypatch.setattr(mod, "get_langfuse_client", lambda: None)
    assert mod.run() == 1
