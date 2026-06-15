"""scripts/eval_push_dataset.py — 골든셋 → Langfuse Dataset 업로드 (#139)."""

import importlib.util
from pathlib import Path

from app.eval.dataset import GoldenQuery


def _load_mod():
    path = Path(__file__).resolve().parents[2] / "scripts" / "eval_push_dataset.py"
    spec = importlib.util.spec_from_file_location("eval_push_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_push_disabled_returns_1(monkeypatch):
    mod = _load_mod()
    monkeypatch.setattr(mod, "get_langfuse_client", lambda: None)
    assert mod.push(Path("q"), Path("r")) == 1


def test_push_uploads_items_idempotent(monkeypatch):
    mod = _load_mod()
    items: list = []
    ds: dict = {}

    class FakeClient:
        def create_dataset(self, **kw):
            ds.update(kw)

        def create_dataset_item(self, **kw):
            items.append(kw)

        def flush(self):
            pass

    monkeypatch.setattr(mod, "get_langfuse_client", lambda: FakeClient())
    golden = [
        GoldenQuery(
            qid="q1",
            query="질문",
            domain="manual",
            is_answerable=True,
            relevant_chunk_ids=("c1", "c2"),
            gold_domains=("manual",),
        )
    ]
    monkeypatch.setattr(mod, "load_golden_set", lambda q, r: golden)

    rc = mod.push(Path("q"), Path("r"), dataset_name="ds-test")

    assert rc == 0
    assert ds["name"] == "ds-test"
    assert len(items) == 1
    it = items[0]
    assert it["id"] == "q1"  # qid 멱등 업서트
    assert it["input"]["query"] == "질문"
    assert it["expected_output"]["relevant_chunk_ids"] == ["c1", "c2"]
    assert it["expected_output"]["is_answerable"] is True
